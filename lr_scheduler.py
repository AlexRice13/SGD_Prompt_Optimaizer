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


if __name__ == '__main__':
    """Unit tests for LRScheduler class."""
    import math
    
    print("Running LRScheduler unit tests...")
    
    # Test 1: Basic initialization
    print("\n1. Testing basic initialization...")
    scheduler = LRScheduler(initial_lr=0.1, min_lr=0.01, max_steps=100, warmup_steps=10)
    assert scheduler.initial_lr == 0.1
    assert scheduler.min_lr == 0.01
    assert scheduler.current_step == 0
    print("   ✓ Initialization works")
    
    # Test 2: Warmup phase
    print("\n2. Testing warmup phase...")
    lr_start = scheduler.get_lr(0)
    lr_mid_warmup = scheduler.get_lr(5)
    lr_end_warmup = scheduler.get_lr(10)
    assert lr_start == scheduler.min_lr
    assert scheduler.min_lr < lr_mid_warmup < scheduler.initial_lr
    assert abs(lr_end_warmup - scheduler.initial_lr) < 1e-6
    print(f"   Warmup: {lr_start:.4f} -> {lr_mid_warmup:.4f} -> {lr_end_warmup:.4f} ✓")
    
    # Test 3: Cosine annealing
    print("\n3. Testing cosine annealing...")
    lr_quarter = scheduler.get_lr(35)  # 25% through annealing
    lr_half = scheduler.get_lr(55)     # 50% through annealing
    lr_end = scheduler.get_lr(100)     # End
    assert lr_quarter > lr_half > lr_end
    assert lr_end >= scheduler.min_lr
    print(f"   Annealing: {lr_quarter:.4f} -> {lr_half:.4f} -> {lr_end:.4f} ✓")
    
    # Test 4: Step function
    print("\n4. Testing step function...")
    scheduler2 = LRScheduler(initial_lr=0.1, min_lr=0.01, max_steps=100, warmup_steps=10)
    lr1 = scheduler2.step()
    lr2 = scheduler2.step()
    assert scheduler2.current_step == 2
    assert lr2 != lr1  # LR should change
    print(f"   Step 1: {lr1:.4f}, Step 2: {lr2:.4f} ✓")
    
    # Test 5: get_current_lr
    print("\n5. Testing get_current_lr...")
    current_lr = scheduler2.get_current_lr()
    assert current_lr == scheduler2.get_lr(scheduler2.current_step)
    print(f"   Current LR: {current_lr:.4f} ✓")
    
    # Test 6: Reset
    print("\n6. Testing reset...")
    scheduler2.reset()
    assert scheduler2.current_step == 0
    assert scheduler2.get_current_lr() == scheduler2.min_lr
    print("   Reset works ✓")
    
    # Test 7: Edge cases
    print("\n7. Testing edge cases...")
    # Zero warmup
    scheduler3 = LRScheduler(initial_lr=0.1, min_lr=0.01, max_steps=100, warmup_steps=0)
    lr_zero = scheduler3.get_lr(0)
    assert abs(lr_zero - scheduler3.initial_lr) < 1e-6
    # Beyond max_steps
    lr_beyond = scheduler3.get_lr(150)
    assert lr_beyond >= scheduler3.min_lr
    print("   Edge cases handled correctly ✓")
    
    print("\n" + "="*50)
    print("All LRScheduler tests passed! ✓")
    print("="*50)
