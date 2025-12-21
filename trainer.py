"""
Trainer: Main SGD training loop for prompt optimization.

Integrates all components to implement the complete optimization framework.
"""

from typing import List, Dict, Callable, Tuple, Optional
import numpy as np
from pathlib import Path
import json

from judge_prompt import JudgePrompt
from forward_pass import ForwardPass
from loss_functions import LossFunctions
from gradient_agent import GradientAgent
from optimizer import PromptOptimizer
from lr_scheduler import LRScheduler
from batch_sampler import BatchSampler
from metrics import Metrics
from early_stopping import EarlyStopping
from version_control import VersionControl


class SGDPromptTrainer:
    """
    SGD-based prompt optimization trainer.
    
    Implements the complete training pipeline for optimizing JudgePrompt
    using SGD algorithm concepts in natural language parameter space.
    """
    
    def __init__(self,
                 judge_llm_fn: Callable[[str, str], float],
                 gradient_llm_fn: Callable[[str], str],
                 optimizer_llm_fn: Callable[[str], str],
                 initial_prompt: JudgePrompt,
                 train_responses: List[str],
                 train_human_scores: np.ndarray,
                 val_responses: List[str],
                 val_human_scores: np.ndarray,
                 config: Optional[Dict] = None):
        """
        Initialize trainer.
        
        Args:
            judge_llm_fn: Function for scoring responses (prompt, response) -> score
            gradient_llm_fn: Function for gradient agent LLM (prompt) -> text
            optimizer_llm_fn: Function for optimizer LLM (prompt) -> text
            initial_prompt: Initial JudgePrompt
            train_responses: Training responses
            train_human_scores: Human scores for training
            val_responses: Validation responses
            val_human_scores: Human scores for validation
            config: Configuration dictionary
        """
        self.judge_llm_fn = judge_llm_fn
        self.gradient_llm_fn = gradient_llm_fn
        self.optimizer_llm_fn = optimizer_llm_fn
        
        # Data
        self.train_responses = train_responses
        self.train_human_scores = train_human_scores
        self.val_responses = val_responses
        self.val_human_scores = val_human_scores
        
        # Prompt
        self.current_prompt = initial_prompt
        
        # Configuration
        self.config = config or {}
        self._setup_config()
        
        # Components
        self._setup_components()
        
        # Training state
        self.current_step = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
    
    def _setup_config(self):
        """Setup configuration with defaults."""
        defaults = {
            'max_steps': 100,
            'batch_size': 32,
            'initial_lr': 0.1,
            'min_lr': 0.001,
            'warmup_steps': 10,
            'alpha': 1.0,  # MAE weight
            'beta': 1.0,   # Rank loss weight
            'loss_type': 'pairwise',
            'n_consistency_samples': 1,
            'n_gradient_samples': 3,
            'patience': 5,
            'checkpoint_dir': './checkpoints',
            'enable_version_control': True,
            'logging_steps': 1,  # Log every N steps (TRL-style)
            'eval_steps': 1,     # Evaluate every N steps (TRL-style)
            'max_workers': 10,   # Max concurrent threads for LLM calls
            'structural_edit_threshold_ratio': 0.5,  # Ratio above which structural edits allowed
            'base_char_limit': 300,  # Base character limit for modifications at initial LR
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _setup_components(self):
        """Initialize all components."""
        # Forward pass
        self.forward_pass = ForwardPass(
            self.judge_llm_fn,
            n_consistency_samples=self.config['n_consistency_samples'],
            max_workers=self.config['max_workers']
        )
        
        # Loss functions
        self.loss_fn = LossFunctions(
            alpha=self.config['alpha'],
            beta=self.config['beta']
        )
        
        # Gradient agent
        self.gradient_agent = GradientAgent(
            self.gradient_llm_fn,
            n_samples_per_category=self.config['n_gradient_samples']
        )
        
        # Optimizer
        self.optimizer = PromptOptimizer(
            self.optimizer_llm_fn,
            structural_edit_threshold_ratio=self.config['structural_edit_threshold_ratio'],
            initial_lr=self.config['initial_lr'],
            base_char_limit=self.config['base_char_limit']
        )
        
        # Learning rate scheduler
        self.lr_scheduler = LRScheduler(
            initial_lr=self.config['initial_lr'],
            min_lr=self.config['min_lr'],
            max_steps=self.config['max_steps'],
            warmup_steps=self.config['warmup_steps']
        )
        
        # Batch sampler
        self.batch_sampler = BatchSampler(
            batch_size=self.config['batch_size']
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['patience']
        )
        
        # Version control
        if self.config['enable_version_control']:
            checkpoint_dir = Path(self.config['checkpoint_dir'])
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            self.version_control = VersionControl(str(checkpoint_dir))
        else:
            self.version_control = None
    
    def evaluate(self, responses: List[str], human_scores: np.ndarray) -> Tuple[float, Dict]:
        """
        Evaluate current prompt on a dataset.
        
        Args:
            responses: List of responses
            human_scores: Human reference scores
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Score all responses
        prompt_text = self.current_prompt.get_full_prompt()
        score_results = self.forward_pass.score_batch(prompt_text, responses)
        
        judge_scores = self.forward_pass.get_scores_array(score_results)
        variances = self.forward_pass.get_variances_array(score_results)
        
        # Compute loss
        total_loss, mae, rank_loss = self.loss_fn.compute_loss(
            judge_scores, human_scores, self.config['loss_type']
        )
        
        # Compute metrics
        metrics = Metrics.compute_all_metrics(judge_scores, human_scores, variances)
        metrics['mae_component'] = mae
        metrics['rank_loss_component'] = rank_loss
        
        return total_loss, metrics
    
    def train_step(self) -> Dict:
        """
        Execute one training step.
        
        Returns:
            Dictionary with step information
        """
        # Sample batch
        batch_indices, batch_responses = self.batch_sampler.sample_batch(
            self.train_human_scores, self.train_responses
        )
        batch_human_scores = self.train_human_scores[batch_indices]
        
        # Forward pass
        prompt_text = self.current_prompt.get_full_prompt()
        score_results = self.forward_pass.score_batch(prompt_text, batch_responses)
        judge_scores = self.forward_pass.get_scores_array(score_results)
        
        # Compute loss
        total_loss, mae, rank_loss = self.loss_fn.compute_loss(
            judge_scores, batch_human_scores, self.config['loss_type']
        )
        
        # Get learning rate
        lr = self.lr_scheduler.get_current_lr()
        
        # Get editable and meta sections
        editable_sections = list(self.current_prompt.get_editable_sections())
        meta_sections = list(self.current_prompt.meta_sections)
        
        # Compute gradient with structured output
        gradient_result = self.gradient_agent.compute_gradient(
            prompt_text,
            editable_sections,
            meta_sections,
            judge_scores,
            batch_human_scores,
            batch_responses,
            lr,
            self.optimizer.structural_edit_threshold
        )
        
        # Generate modification from structured gradient
        modification_suggestion = self.optimizer.generate_modification_from_structured_gradient(
            prompt_text,
            gradient_result['structured_gradient'],
            lr,
            editable_sections,
            meta_sections
        )
        
        # Parse and validate modification
        modification = self.optimizer.parse_modification(modification_suggestion)
        
        step_info = {
            'step': self.current_step,
            'train_loss': total_loss,
            'mae': mae,
            'rank_loss': rank_loss,
            'learning_rate': lr,
            'gradient_statistics': gradient_result['statistics'],
            'modification_valid': False
        }
        
        if modification:
            print(f"\n=== Modification Validation ===")
            print(f"Target section: {modification['section']}")
            print(f"Is editable: {modification['section'] in editable_sections}")
            print(f"Is meta: {modification['section'] in meta_sections}")
            
            is_valid = self.optimizer.validate_modification(
                modification, lr, editable_sections, meta_sections
            )
            print(f"Validation result: {is_valid}")
            
            if is_valid:
                # Apply modification
                print(f"Attempting to update section: {modification['section']}")
                success = self.current_prompt.update_section(
                    modification['section'],
                    modification['new_content']
                )
                print(f"Update success: {success}")
                
                if success:
                    step_info['modification_valid'] = True
                    step_info['modified_section'] = modification['section']
                    step_info['modification_rationale'] = modification.get('rationale', '')
                    print(f"✓ Successfully modified section: {modification['section']}")
                else:
                    print(f"✗ Failed to update section (might be meta): {modification['section']}")
            else:
                print(f"✗ Modification validation failed")
        else:
            print("✗ Failed to parse modification from LLM output")
        
        return step_info
    
    def train(self) -> JudgePrompt:
        """
        Run full training loop with configurable logging and evaluation.
        
        Returns:
            Best JudgePrompt found during training
        """
        print(f"Starting training for {self.config['max_steps']} steps...")
        print(f"Train samples: {len(self.train_responses)}, Val samples: {len(self.val_responses)}")
        print(f"Logging every {self.config['logging_steps']} steps, "
              f"evaluating every {self.config['eval_steps']} steps")
        
        # Initial evaluation
        val_loss, val_metrics = self.evaluate(self.val_responses, self.val_human_scores)
        train_loss, train_metrics = self.evaluate(self.train_responses, self.train_human_scores)
        
        print(f"\n{'='*80}")
        print(f"Initial Evaluation")
        print(f"  Train Loss: {train_loss:.4f}, MAE: {train_metrics['mae']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.4f}, "
              f"Kendall τ: {val_metrics['kendall_tau']:.4f}")
        print(f"{'='*80}")
        
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_metrics'].append(train_metrics)
        self.history['val_metrics'].append(val_metrics)
        self.history['learning_rates'].append(0.0)
        
        # Save initial prompt
        if self.version_control:
            self.current_prompt.save(self.version_control.prompt_path)
            self.version_control.commit_prompt_update(
                0, train_loss, val_loss, val_metrics,
                "Initial prompt", "Baseline"
            )
        
        best_prompt = JudgePrompt.from_dict(self.current_prompt.to_dict())
        best_val_loss = val_loss
        
        for step in range(self.config['max_steps']):
            self.current_step = step + 1
            
            # Training step
            step_info = self.train_step()
            
            # Determine if we should evaluate this step
            should_eval = (self.current_step % self.config['eval_steps'] == 0) or \
                         (self.current_step == self.config['max_steps'])
            
            # Determine if we should log this step
            should_log = (self.current_step % self.config['logging_steps'] == 0) or \
                        (self.current_step == self.config['max_steps'])
            
            # Evaluation (only on eval_steps)
            if should_eval:
                val_loss, val_metrics = self.evaluate(self.val_responses, self.val_human_scores)
                train_loss, train_metrics = self.evaluate(self.train_responses, self.train_human_scores)
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_metrics'].append(train_metrics)
                self.history['val_metrics'].append(val_metrics)
                self.history['learning_rates'].append(step_info['learning_rate'])
            else:
                # Use step loss for quick feedback (not full evaluation)
                train_loss = step_info['train_loss']
                val_loss = None
                train_metrics = None
                val_metrics = None
            
            # Logging (TRL-style)
            if should_log:
                epoch = self.batch_sampler.get_current_epoch()
                epoch_progress = self.batch_sampler.get_epoch_progress()
                
                print(f"\n{'='*80}")
                print(f"Step {self.current_step}/{self.config['max_steps']} "
                      f"[Epoch {epoch}, {epoch_progress:.1%} complete]")
                print(f"  LR: {step_info['learning_rate']:.4f}")
                print(f"  Step Train Loss: {step_info['train_loss']:.4f} "
                      f"(MAE: {step_info['mae']:.4f}, Rank: {step_info['rank_loss']:.4f})")
                
                if should_eval and val_loss is not None:
                    print(f"  Full Train Loss: {train_loss:.4f} (MAE: {train_metrics['mae']:.4f}, "
                          f"Rank: {train_metrics['rank_loss_component']:.4f})")
                    print(f"  Val Loss: {val_loss:.4f} (MAE: {val_metrics['mae']:.4f}, "
                          f"Kendall τ: {val_metrics['kendall_tau']:.4f})")
                
                if step_info['modification_valid']:
                    print(f"  ✓ Modified section: {step_info['modified_section']}")
                else:
                    print(f"  ✗ No valid modification applied")
                print(f"{'='*80}")
            
            # Save checkpoint (only when modification is valid and evaluation is performed)
            if self.version_control and step_info['modification_valid'] and should_eval and val_loss is not None:
                self.current_prompt.save(self.version_control.prompt_path)
                self.version_control.commit_prompt_update(
                    self.current_step,
                    train_loss,
                    val_loss,
                    val_metrics,
                    str(step_info['gradient_statistics']),
                    step_info.get('modification_rationale', '')
                )
            
            # Update best prompt (only on evaluation steps when val_loss is available)
            if should_eval and val_loss is not None and val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_prompt = JudgePrompt.from_dict(self.current_prompt.to_dict())
                if self.version_control:
                    self.version_control.create_checkpoint_tag(self.current_step, is_best=True)
            
            # Check early stopping (only on evaluation steps when val_loss is available)
            if should_eval and val_loss is not None and self.early_stopping.step(val_loss, val_metrics, self.current_step):
                print(f"\n{'='*80}")
                print(f"Early stopping triggered at step {self.current_step}")
                print(f"{'='*80}")
                break
            
            # Update learning rate
            self.lr_scheduler.step()
        
        print(f"\n{'='*80}")
        print(f"Training Completed")
        print(f"  Best step: {self.early_stopping.get_best_step()}")
        print(f"  Best val loss: {self.early_stopping.get_best_loss():.4f}")
        print(f"{'='*80}")
        
        return best_prompt
    
    def save_history(self, filepath: str):
        """Save training history to JSON file."""
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_serializable = {
                'train_loss': [float(x) for x in self.history['train_loss']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'learning_rates': [float(x) for x in self.history['learning_rates']],
                'train_metrics': self.history['train_metrics'],
                'val_metrics': self.history['val_metrics'],
            }
            json.dump(history_serializable, f, indent=2)
