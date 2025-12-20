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
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _setup_components(self):
        """Initialize all components."""
        # Forward pass
        self.forward_pass = ForwardPass(
            self.judge_llm_fn,
            n_consistency_samples=self.config['n_consistency_samples']
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
        self.optimizer = PromptOptimizer(self.optimizer_llm_fn)
        
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
        
        # Compute gradient
        gradient_result = self.gradient_agent.compute_gradient(
            prompt_text, judge_scores, batch_human_scores
        )
        
        # Get learning rate
        lr = self.lr_scheduler.get_current_lr()
        
        # Get editable and meta sections
        editable_sections = list(self.current_prompt.get_editable_sections())
        meta_sections = list(self.current_prompt.meta_sections)
        
        # Generate modification suggestion
        modification_suggestion = self.optimizer.generate_modification_suggestion(
            prompt_text,
            gradient_result['proxy_gradient'],
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
        
        if modification and self.optimizer.validate_modification(
            modification, lr, editable_sections, meta_sections
        ):
            # Apply modification
            success = self.current_prompt.update_section(
                modification['section'],
                modification['new_content']
            )
            
            if success:
                step_info['modification_valid'] = True
                step_info['modified_section'] = modification['section']
                step_info['modification_rationale'] = modification.get('rationale', '')
        
        return step_info
    
    def train(self) -> JudgePrompt:
        """
        Run full training loop.
        
        Returns:
            Best JudgePrompt found during training
        """
        print(f"Starting training for {self.config['max_steps']} steps...")
        print(f"Train samples: {len(self.train_responses)}, Val samples: {len(self.val_responses)}")
        
        # Initial evaluation
        val_loss, val_metrics = self.evaluate(self.val_responses, self.val_human_scores)
        print(f"\nInitial validation - Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.4f}, "
              f"Kendall τ: {val_metrics['kendall_tau']:.4f}")
        
        # Save initial prompt
        if self.version_control:
            self.current_prompt.save(self.version_control.prompt_path)
            self.version_control.commit_prompt_update(
                0, 0.0, val_loss, val_metrics,
                "Initial prompt", "Baseline"
            )
        
        best_prompt = JudgePrompt.from_dict(self.current_prompt.to_dict())
        
        for step in range(self.config['max_steps']):
            self.current_step = step + 1
            
            # Training step
            step_info = self.train_step()
            
            # Evaluation
            val_loss, val_metrics = self.evaluate(self.val_responses, self.val_human_scores)
            train_loss, train_metrics = self.evaluate(self.train_responses, self.train_human_scores)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(step_info['learning_rate'])
            
            # Log
            print(f"\nStep {self.current_step}/{self.config['max_steps']}")
            print(f"  LR: {step_info['learning_rate']:.4f}")
            print(f"  Train Loss: {train_loss:.4f} (MAE: {train_metrics['mae']:.4f}, "
                  f"Rank: {train_metrics['rank_loss_component']:.4f})")
            print(f"  Val Loss: {val_loss:.4f} (MAE: {val_metrics['mae']:.4f}, "
                  f"Kendall τ: {val_metrics['kendall_tau']:.4f})")
            
            if step_info['modification_valid']:
                print(f"  Modified section: {step_info['modified_section']}")
            
            # Save checkpoint
            if self.version_control and step_info['modification_valid']:
                self.current_prompt.save(self.version_control.prompt_path)
                self.version_control.commit_prompt_update(
                    self.current_step,
                    train_loss,
                    val_loss,
                    val_metrics,
                    str(step_info['gradient_statistics']),
                    step_info.get('modification_rationale', '')
                )
            
            # Update best prompt if this is the best so far (before early stopping check)
            if val_loss <= self.early_stopping.get_best_loss():
                best_prompt = JudgePrompt.from_dict(self.current_prompt.to_dict())
                if self.version_control:
                    self.version_control.create_checkpoint_tag(self.current_step, is_best=True)
            
            # Check early stopping
            if self.early_stopping.step(val_loss, val_metrics, self.current_step):
                print(f"\nEarly stopping triggered at step {self.current_step}")
                break
            
            # Update learning rate
            self.lr_scheduler.step()
        
        print(f"\nTraining completed. Best step: {self.early_stopping.get_best_step()}")
        print(f"Best val loss: {self.early_stopping.get_best_loss():.4f}")
        
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
