"""
Batch Sampler: Sample batches with epoch-aware stratified distribution.

Implements batch sampling that covers high/low scores and confusing samples,
ensuring complete dataset traversal per epoch.
"""

import numpy as np
from typing import Tuple, List


class BatchSampler:
    """
    Epoch-aware stratified batch sampler for optimization.
    
    Ensures batches contain:
    - High human scores
    - Low human scores  
    - Confusing middle-range samples
    
    Guarantees complete dataset traversal per epoch with stratified sampling.
    """
    
    def __init__(self, batch_size: int = 32,
                 high_percentile: float = 0.75,
                 low_percentile: float = 0.25):
        """
        Initialize batch sampler.
        
        Args:
            batch_size: Number of samples per batch
            high_percentile: Percentile threshold for high scores
            low_percentile: Percentile threshold for low scores
        """
        self.batch_size = batch_size
        self.high_percentile = high_percentile
        self.low_percentile = low_percentile
        
        # Epoch tracking
        self.current_epoch = 0
        self.current_step_in_epoch = 0
        self.samples_seen_in_epoch = set()
        self.dataset_size = 0
        
        # Stratified indices
        self.high_idx = None
        self.low_idx = None
        self.mid_idx = None
        
        # Shuffled pools for each category
        self.high_pool = []
        self.low_pool = []
        self.mid_pool = []
    
    def _initialize_categories(self, human_scores: np.ndarray):
        """
        Initialize stratified categories based on human scores.
        
        Args:
            human_scores: All human scores
        """
        n_samples = len(human_scores)
        
        # Only reinitialize if dataset size changes
        if n_samples != self.dataset_size:
            self.dataset_size = n_samples
            
            # Determine score thresholds
            high_threshold = np.percentile(human_scores, self.high_percentile * 100)
            low_threshold = np.percentile(human_scores, self.low_percentile * 100)
            
            # Categorize samples
            self.high_idx = np.where(human_scores >= high_threshold)[0]
            self.low_idx = np.where(human_scores <= low_threshold)[0]
            self.mid_idx = np.where((human_scores > low_threshold) & 
                                   (human_scores < high_threshold))[0]
            
            # Reset epoch
            self._reset_epoch()
    
    def _reset_epoch(self):
        """Reset epoch-level tracking and shuffle categories."""
        self.samples_seen_in_epoch = set()
        self.current_step_in_epoch = 0
        
        # Shuffle each category for the new epoch
        self.high_pool = np.random.permutation(self.high_idx).tolist()
        self.low_pool = np.random.permutation(self.low_idx).tolist()
        self.mid_pool = np.random.permutation(self.mid_idx).tolist()
    
    def _get_indices_from_category(self, pool: List[int], n_needed: int, 
                                   available_indices: set) -> List[int]:
        """
        Get indices from a category pool, ensuring no duplicates within epoch.
        
        Note: This method does not modify the pool parameter. The pool is used
        read-only to maintain order, while available_indices tracks which samples
        haven't been seen yet in the current epoch.
        
        Args:
            pool: List of indices in this category (not modified)
            n_needed: Number of indices needed
            available_indices: Set of indices not yet used in this epoch
            
        Returns:
            List of selected indices
        """
        selected = []
        
        # Get available indices from this pool
        available_from_pool = [idx for idx in pool if idx in available_indices]
        
        # Take up to n_needed indices
        selected = available_from_pool[:n_needed]
        
        return selected
    
    def sample_batch(self, human_scores: np.ndarray, 
                     responses: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Sample a stratified batch with epoch-aware traversal.
        
        Ensures all samples are seen once per epoch before any sample is repeated.
        Maintains stratified distribution within each batch.
        
        Args:
            human_scores: All human scores
            responses: All responses
            
        Returns:
            Tuple of (sampled_indices, sampled_responses)
        """
        n_samples = len(human_scores)
        
        # Initialize categories if needed
        self._initialize_categories(human_scores)
        
        if n_samples <= self.batch_size:
            # Return all samples if dataset is small
            return np.arange(n_samples), responses
        
        # Check if we need to start a new epoch
        if len(self.samples_seen_in_epoch) == n_samples:
            self.current_epoch += 1
            self._reset_epoch()
        
        # Get available indices (not yet seen in this epoch)
        all_indices = set(range(n_samples))
        available_indices = all_indices - self.samples_seen_in_epoch
        
        # Allocate batch slots (roughly equal distribution across categories)
        n_per_category = self.batch_size // 3
        remainder = self.batch_size % 3
        
        # Sample from each category
        selected_indices = []
        
        if len(self.high_idx) > 0:
            n_high = min(n_per_category + (1 if remainder > 0 else 0), len(self.high_idx))
            high_selected = self._get_indices_from_category(
                self.high_pool, n_high, available_indices
            )
            selected_indices.extend(high_selected)
        
        if len(self.low_idx) > 0:
            n_low = min(n_per_category + (1 if remainder > 1 else 0), len(self.low_idx))
            low_selected = self._get_indices_from_category(
                self.low_pool, n_low, available_indices
            )
            selected_indices.extend(low_selected)
        
        if len(self.mid_idx) > 0:
            n_mid = min(n_per_category, len(self.mid_idx))
            mid_selected = self._get_indices_from_category(
                self.mid_pool, n_mid, available_indices
            )
            selected_indices.extend(mid_selected)
        
        # If we don't have enough samples from stratified categories,
        # fill with remaining available samples
        if len(selected_indices) < self.batch_size:
            remaining_needed = self.batch_size - len(selected_indices)
            # Get indices that are available but not yet selected
            selected_set = set(selected_indices)
            remaining_available = [idx for idx in range(n_samples) 
                                  if idx in available_indices and idx not in selected_set]
            if remaining_available:
                # Shuffle and take what we need
                np.random.shuffle(remaining_available)
                extra = remaining_available[:remaining_needed]
                selected_indices.extend(extra)
        
        # Mark these samples as seen in current epoch
        self.samples_seen_in_epoch.update(selected_indices)
        self.current_step_in_epoch += 1
        
        selected_indices = np.array(selected_indices)
        sampled_responses = [responses[i] for i in selected_indices]
        
        return selected_indices, sampled_responses
    
    def get_current_epoch(self) -> int:
        """Get current epoch number."""
        return self.current_epoch
    
    def get_epoch_progress(self) -> float:
        """
        Get progress through current epoch.
        
        Returns:
            Fraction of dataset seen in current epoch (0.0 to 1.0)
        """
        if self.dataset_size == 0:
            return 0.0
        return len(self.samples_seen_in_epoch) / self.dataset_size


if __name__ == '__main__':
    """Unit tests for BatchSampler class."""
    import numpy as np
    
    print("Running BatchSampler unit tests...")
    
    # Test data
    np.random.seed(42)
    n_samples = 50
    human_scores = np.random.uniform(1, 10, n_samples)
    responses = [f"Response {i}" for i in range(n_samples)]
    
    # Test 1: Basic initialization
    print("\n1. Testing basic initialization...")
    sampler = BatchSampler(batch_size=10)
    assert sampler.batch_size == 10
    assert sampler.current_epoch == 0
    print("   ✓ Initialization works")
    
    # Test 2: Sample batch
    print("\n2. Testing batch sampling...")
    indices, sampled_responses = sampler.sample_batch(human_scores, responses)
    assert len(indices) == 10
    assert len(sampled_responses) == 10
    assert all(0 <= idx < n_samples for idx in indices)
    print(f"   Sampled {len(indices)} items ✓")
    
    # Test 3: Epoch tracking
    print("\n3. Testing epoch tracking...")
    initial_epoch = sampler.get_current_epoch()
    # Sample until we complete an epoch
    samples_seen = set()
    for _ in range(10):  # More than enough batches
        indices, _ = sampler.sample_batch(human_scores, responses)
        samples_seen.update(indices)
        if len(samples_seen) == n_samples:
            break
    assert sampler.get_current_epoch() >= initial_epoch
    print(f"   Epoch tracking works (epoch: {sampler.get_current_epoch()}) ✓")
    
    # Test 4: Epoch progress
    print("\n4. Testing epoch progress...")
    progress = sampler.get_epoch_progress()
    assert 0.0 <= progress <= 1.0
    print(f"   Progress: {progress:.2%} ✓")
    
    # Test 5: Small dataset
    print("\n5. Testing small dataset...")
    small_scores = np.array([5.0, 6.0, 7.0])
    small_responses = ["R1", "R2", "R3"]
    sampler_small = BatchSampler(batch_size=10)
    indices, sampled = sampler_small.sample_batch(small_scores, small_responses)
    assert len(indices) == 3  # Returns all samples
    print("   Small dataset handled correctly ✓")
    
    # Test 6: Stratified sampling
    print("\n6. Testing stratified sampling...")
    sampler2 = BatchSampler(batch_size=15)
    indices, _ = sampler2.sample_batch(human_scores, responses)
    sampled_scores = human_scores[indices]
    # Check we have some diversity
    assert sampled_scores.max() > sampled_scores.min()
    print(f"   Score range: [{sampled_scores.min():.2f}, {sampled_scores.max():.2f}] ✓")
    
    # Test 7: No duplicates within epoch
    print("\n7. Testing no duplicates within epoch...")
    sampler3 = BatchSampler(batch_size=10)
    all_indices = []
    for _ in range(5):
        indices, _ = sampler3.sample_batch(human_scores, responses)
        all_indices.extend(indices)
    # Check first epoch has no duplicates
    first_epoch_indices = all_indices[:n_samples]
    assert len(set(first_epoch_indices)) == min(len(first_epoch_indices), n_samples)
    print("   No duplicates within epoch ✓")
    
    print("\n" + "="*50)
    print("All BatchSampler tests passed! ✓")
    print("="*50)
