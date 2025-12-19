"""
Learning Rate Scheduler: Cosine annealing with warmup.

Implements learning rate scheduling for the optimization process.
"""

import math
from typing import Optional


class LRScheduler:
    """
    Learning rate scheduler with cosine annealing and warmup.
    """
    
    def __init__(self, 
                 initial_lr: float = 0.1,
                 min_lr: float = 0.001,
                 max_steps: int = 100,
                 warmup_steps: int = 10):
        """
        Initialize learning rate scheduler.
        
        Args:
            initial_lr: Initial learning rate (peak after warmup)
            min_lr: Minimum learning rate
            max_steps: Total number of optimization steps
            warmup_steps: Number of warmup steps
        """
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def get_lr(self, step: Optional[int] = None) -> float:
        """
        Get learning rate for a specific step.
        
        Args:
            step: Step number (uses current_step if None)
            
        Returns:
            Learning rate value
        """
        if step is None:
            step = self.current_step
        
        # Warmup phase
        if step < self.warmup_steps:
            return self.min_lr + (self.initial_lr - self.min_lr) * (step / self.warmup_steps)
        
        # Cosine annealing phase
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        return lr
    
    def step(self) -> float:
        """
        Advance to next step and return new learning rate.
        
        Returns:
            Learning rate for new step
        """
        self.current_step += 1
        return self.get_lr()
    
    def get_current_lr(self) -> float:
        """Get learning rate for current step."""
        return self.get_lr()
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0
