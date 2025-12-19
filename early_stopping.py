"""
Early Stopping: Monitor validation loss and prevent overfitting.

Implements early stopping with multiple monitoring criteria:
- Validation loss
- Rank correlation
- Score entropy (collapse detection)
- Self-consistency variance
"""

from typing import Optional, Dict
import numpy as np


class EarlyStopping:
    """
    Early stopping with multiple monitoring criteria.
    """
    
    def __init__(self, 
                 patience: int = 5,
                 min_delta: float = 0.0,
                 monitor_rank_corr: bool = True,
                 monitor_entropy: bool = True,
                 monitor_variance: bool = True,
                 entropy_threshold: float = 0.5,
                 variance_threshold: float = 0.1):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of steps to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor_rank_corr: Monitor rank correlation decline
            monitor_entropy: Monitor score entropy collapse
            monitor_variance: Monitor self-consistency variance increase
            entropy_threshold: Threshold below which entropy triggers warning
            variance_threshold: Threshold above which variance triggers warning
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_rank_corr = monitor_rank_corr
        self.monitor_entropy = monitor_entropy
        self.monitor_variance = monitor_variance
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        
        self.best_loss = np.inf
        self.best_step = 0
        self.counter = 0
        self.should_stop = False
        
        self.loss_history = []
        self.rank_corr_history = []
        self.entropy_history = []
        self.variance_history = []
    
    def step(self, val_loss: float, metrics: Dict[str, float], 
             current_step: int) -> bool:
        """
        Update early stopping state.
        
        Args:
            val_loss: Validation loss
            metrics: Dictionary of metrics
            current_step: Current training step
            
        Returns:
            True if training should stop
        """
        self.loss_history.append(val_loss)
        
        # Track metrics
        if 'kendall_tau' in metrics:
            self.rank_corr_history.append(metrics['kendall_tau'])
        if 'score_entropy' in metrics:
            self.entropy_history.append(metrics['score_entropy'])
        if 'mean_self_consistency_var' in metrics:
            self.variance_history.append(metrics['mean_self_consistency_var'])
        
        # Check for improvement in validation loss
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_step = current_step
            self.counter = 0
        else:
            self.counter += 1
        
        # Check secondary criteria
        warnings = []
        
        if self.monitor_rank_corr and len(self.rank_corr_history) >= 3:
            # Check if rank correlation is declining
            recent_corr = np.mean(self.rank_corr_history[-3:])
            if len(self.rank_corr_history) > 5:
                earlier_corr = np.mean(self.rank_corr_history[-6:-3])
                if recent_corr < earlier_corr - 0.05:
                    warnings.append("rank_correlation_decline")
        
        if self.monitor_entropy and len(self.entropy_history) >= 2:
            # Check if score entropy is collapsing
            recent_entropy = self.entropy_history[-1]
            if recent_entropy < self.entropy_threshold:
                warnings.append("entropy_collapse")
        
        if self.monitor_variance and len(self.variance_history) >= 2:
            # Check if self-consistency variance is increasing
            recent_variance = self.variance_history[-1]
            if recent_variance > self.variance_threshold:
                warnings.append("variance_increase")
        
        # Trigger early stopping if patience exceeded
        if self.counter >= self.patience:
            self.should_stop = True
        
        # Also trigger if we have critical warnings
        if len(warnings) >= 2:  # Multiple warnings
            self.should_stop = True
        
        return self.should_stop
    
    def get_best_step(self) -> int:
        """Get the step with best validation loss."""
        return self.best_step
    
    def get_best_loss(self) -> float:
        """Get the best validation loss observed."""
        return self.best_loss
    
    def reset(self):
        """Reset early stopping state."""
        self.best_loss = np.inf
        self.best_step = 0
        self.counter = 0
        self.should_stop = False
        self.loss_history = []
        self.rank_corr_history = []
        self.entropy_history = []
        self.variance_history = []
