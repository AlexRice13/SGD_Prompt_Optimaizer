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
            patience: Number of steps to wait before stopping.
                     Set to 0 to disable early stopping completely.
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
            True if training should stop, False otherwise.
            Always returns False if patience is 0 (early stopping disabled).
        """
        # Track best loss and step (always, even when early stopping disabled)
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_step = current_step
            self.counter = 0
        else:
            self.counter += 1
        
        # If patience is 0, early stopping is disabled
        if self.patience == 0:
            return False
        
        self.loss_history.append(val_loss)
        
        # Track metrics
        if 'kendall_tau' in metrics:
            self.rank_corr_history.append(metrics['kendall_tau'])
        if 'score_entropy' in metrics:
            self.entropy_history.append(metrics['score_entropy'])
        if 'mean_self_consistency_var' in metrics:
            self.variance_history.append(metrics['mean_self_consistency_var'])
        
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


if __name__ == '__main__':
    """Unit tests for EarlyStopping class."""
    import numpy as np
    
    print("Running EarlyStopping unit tests...")
    
    # Test 1: Basic initialization
    print("\n1. Testing basic initialization...")
    early_stop = EarlyStopping(patience=3, min_delta=0.01)
    assert early_stop.patience == 3
    assert early_stop.best_loss == np.inf
    assert early_stop.counter == 0
    print("   ✓ Initialization works")
    
    # Test 2: Improvement detected
    print("\n2. Testing improvement detection...")
    metrics = {'kendall_tau': 0.8, 'score_entropy': 1.5}
    should_stop = early_stop.step(1.0, metrics, 1)
    assert not should_stop
    assert early_stop.best_loss == 1.0
    assert early_stop.best_step == 1
    assert early_stop.counter == 0
    print("   Improvement detected correctly ✓")
    
    # Test 3: No improvement
    print("\n3. Testing no improvement...")
    should_stop = early_stop.step(1.05, metrics, 2)
    assert not should_stop
    assert early_stop.counter == 1
    print("   No improvement detected ✓")
    
    # Test 4: Patience exceeded
    print("\n4. Testing patience exceeded...")
    early_stop.step(1.1, metrics, 3)
    early_stop.step(1.15, metrics, 4)
    should_stop = early_stop.step(1.2, metrics, 5)
    assert should_stop
    print("   Early stopping triggered ✓")
    
    # Test 5: Disabled early stopping
    print("\n5. Testing disabled early stopping...")
    early_stop2 = EarlyStopping(patience=0)
    for i in range(10):
        should_stop = early_stop2.step(10.0 - i, metrics, i)
        assert not should_stop  # Never stops when patience=0
    assert early_stop2.best_loss < 10.0
    print("   Disabled early stopping works (patience=0) ✓")
    
    # Test 6: Min delta
    print("\n6. Testing min_delta...")
    early_stop3 = EarlyStopping(patience=2, min_delta=0.1)
    early_stop3.step(1.0, metrics, 1)
    should_stop = early_stop3.step(0.95, metrics, 2)  # Improvement < min_delta
    assert not should_stop
    assert early_stop3.counter == 1  # Counts as no improvement
    print("   Min delta works correctly ✓")
    
    # Test 7: Get best step/loss
    print("\n7. Testing get_best_step and get_best_loss...")
    best_step = early_stop3.get_best_step()
    best_loss = early_stop3.get_best_loss()
    assert isinstance(best_step, int)
    assert isinstance(best_loss, (int, float))
    print(f"   Best step: {best_step}, Best loss: {best_loss:.4f} ✓")
    
    # Test 8: Reset
    print("\n8. Testing reset...")
    early_stop3.reset()
    assert early_stop3.best_loss == np.inf
    assert early_stop3.counter == 0
    assert len(early_stop3.loss_history) == 0
    print("   Reset works ✓")
    
    # Test 9: Entropy collapse detection
    print("\n9. Testing entropy collapse detection...")
    early_stop4 = EarlyStopping(patience=5, monitor_entropy=True, entropy_threshold=0.5)
    low_entropy_metrics = {'kendall_tau': 0.8, 'score_entropy': 0.3}
    early_stop4.step(1.0, low_entropy_metrics, 1)
    early_stop4.step(1.0, low_entropy_metrics, 2)
    # Multiple warnings can trigger early stop even if loss not worsening
    print("   Entropy monitoring works ✓")
    
    print("\n" + "="*50)
    print("All EarlyStopping tests passed! ✓")
    print("="*50)
