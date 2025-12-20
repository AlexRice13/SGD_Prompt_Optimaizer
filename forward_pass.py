"""
Forward Pass: Score responses using JudgePrompt.

Implements the forward pass where JudgePrompt is used to score responses,
with support for self-consistency variance estimation and concurrent LLM calls.
"""

from typing import List, Dict, Callable, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


class ForwardPass:
    """
    Forward pass implementation for scoring responses.
    
    Uses a Judge LLM (provided as a callable) to score responses
    with the current JudgePrompt. Supports concurrent LLM calls for efficiency.
    """
    
    def __init__(self, judge_llm_fn: Callable[[str, str], float], 
                 n_consistency_samples: int = 1,
                 max_workers: int = 10):
        """
        Initialize forward pass.
        
        Args:
            judge_llm_fn: Function that takes (prompt, response) and returns a score
            n_consistency_samples: Number of times to score each response for
                                  self-consistency variance estimation
            max_workers: Maximum number of concurrent threads for LLM calls
        """
        self.judge_llm_fn = judge_llm_fn
        self.n_consistency_samples = n_consistency_samples
        self.max_workers = max_workers
    
    def score_single(self, prompt: str, response: str) -> Dict[str, float]:
        """
        Score a single response with self-consistency estimation.
        
        Args:
            prompt: JudgePrompt text
            response: Response to score
            
        Returns:
            Dictionary with 'mean_score' and 'variance'
        """
        if self.n_consistency_samples == 1:
            # No concurrency needed for single sample
            score = self.judge_llm_fn(prompt, response)
            return {
                'mean_score': float(score),
                'variance': 0.0,
                'scores': [score]
            }
        
        # Concurrent scoring for self-consistency
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.judge_llm_fn, prompt, response) 
                      for _ in range(self.n_consistency_samples)]
            scores = [future.result() for future in futures]
        
        return {
            'mean_score': float(np.mean(scores)),
            'variance': float(np.var(scores)) if len(scores) > 1 else 0.0,
            'scores': scores
        }
    
    def score_batch(self, prompt: str, responses: List[str]) -> List[Dict[str, float]]:
        """
        Score a batch of responses with concurrent LLM calls.
        
        Uses map-like pattern to preserve order of results matching input order.
        
        Args:
            prompt: JudgePrompt text
            responses: List of responses to score
            
        Returns:
            List of score dictionaries for each response (order preserved)
        """
        if len(responses) == 0:
            return []
        
        # Use ThreadPoolExecutor to score responses concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks and maintain order using map
            results = list(executor.map(
                lambda response: self.score_single(prompt, response),
                responses
            ))
        
        return results
    
    def get_scores_array(self, score_results: List[Dict[str, float]]) -> np.ndarray:
        """Extract mean scores as numpy array."""
        return np.array([result['mean_score'] for result in score_results])
    
    def get_variances_array(self, score_results: List[Dict[str, float]]) -> np.ndarray:
        """Extract variances as numpy array."""
        return np.array([result['variance'] for result in score_results])
