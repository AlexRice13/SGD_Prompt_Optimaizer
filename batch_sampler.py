"""
Batch Sampler: Sample batches with stratified distribution.

Implements batch sampling that covers high/low scores and confusing samples.
"""

import numpy as np
from typing import Tuple, List


class BatchSampler:
    """
    Stratified batch sampler for optimization.
    
    Ensures batches contain:
    - High human scores
    - Low human scores  
    - Confusing middle-range samples
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
    
    def sample_batch(self, human_scores: np.ndarray, 
                     responses: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Sample a stratified batch.
        
        Args:
            human_scores: All human scores
            responses: All responses
            
        Returns:
            Tuple of (sampled_indices, sampled_responses)
        """
        n_samples = len(human_scores)
        
        if n_samples <= self.batch_size:
            # Return all samples if dataset is small
            return np.arange(n_samples), responses
        
        # Determine score thresholds
        high_threshold = np.percentile(human_scores, self.high_percentile * 100)
        low_threshold = np.percentile(human_scores, self.low_percentile * 100)
        
        # Categorize samples
        high_idx = np.where(human_scores >= high_threshold)[0]
        low_idx = np.where(human_scores <= low_threshold)[0]
        mid_idx = np.where((human_scores > low_threshold) & 
                          (human_scores < high_threshold))[0]
        
        # Allocate batch slots (roughly equal distribution)
        n_per_category = self.batch_size // 3
        remainder = self.batch_size % 3
        
        # Sample from each category
        selected_indices = []
        
        if len(high_idx) > 0:
            n_high = min(n_per_category + (1 if remainder > 0 else 0), len(high_idx))
            selected_indices.extend(np.random.choice(high_idx, n_high, replace=False))
        
        if len(low_idx) > 0:
            n_low = min(n_per_category + (1 if remainder > 1 else 0), len(low_idx))
            selected_indices.extend(np.random.choice(low_idx, n_low, replace=False))
        
        if len(mid_idx) > 0:
            n_mid = min(n_per_category, len(mid_idx))
            selected_indices.extend(np.random.choice(mid_idx, n_mid, replace=False))
        
        # If we don't have enough samples, fill with random samples
        if len(selected_indices) < self.batch_size:
            remaining = self.batch_size - len(selected_indices)
            all_idx = np.arange(n_samples)
            available_idx = np.setdiff1d(all_idx, selected_indices)
            if len(available_idx) > 0:
                extra = np.random.choice(available_idx, 
                                        min(remaining, len(available_idx)), 
                                        replace=False)
                selected_indices.extend(extra)
        
        selected_indices = np.array(selected_indices)
        sampled_responses = [responses[i] for i in selected_indices]
        
        return selected_indices, sampled_responses
