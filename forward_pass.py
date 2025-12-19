"""
Forward Pass: Score responses using JudgePrompt.

Implements the forward pass where JudgePrompt is used to score responses,
with support for self-consistency variance estimation.
"""

from typing import List, Dict, Callable, Optional
import numpy as np


class ForwardPass:
    """
    Forward pass implementation for scoring responses.
    
    Uses a Judge LLM (provided as a callable) to score responses
    with the current JudgePrompt.
    """
    
    def __init__(self, judge_llm_fn: Callable[[str, str], float], 
                 n_consistency_samples: int = 1):
        """
        Initialize forward pass.
        
        Args:
            judge_llm_fn: Function that takes (prompt, response) and returns a score
            n_consistency_samples: Number of times to score each response for
                                  self-consistency variance estimation
        """
        self.judge_llm_fn = judge_llm_fn
        self.n_consistency_samples = n_consistency_samples
    
    def score_single(self, prompt: str, response: str) -> Dict[str, float]:
        """
        Score a single response with self-consistency estimation.
        
        Args:
            prompt: JudgePrompt text
            response: Response to score
            
        Returns:
            Dictionary with 'mean_score' and 'variance'
        """
        scores = []
        for _ in range(self.n_consistency_samples):
            score = self.judge_llm_fn(prompt, response)
            scores.append(score)
        
        return {
            'mean_score': float(np.mean(scores)),
            'variance': float(np.var(scores)) if len(scores) > 1 else 0.0,
            'scores': scores
        }
    
    def score_batch(self, prompt: str, responses: List[str]) -> List[Dict[str, float]]:
        """
        Score a batch of responses.
        
        Args:
            prompt: JudgePrompt text
            responses: List of responses to score
            
        Returns:
            List of score dictionaries for each response
        """
        return [self.score_single(prompt, response) for response in responses]
    
    def get_scores_array(self, score_results: List[Dict[str, float]]) -> np.ndarray:
        """Extract mean scores as numpy array."""
        return np.array([result['mean_score'] for result in score_results])
    
    def get_variances_array(self, score_results: List[Dict[str, float]]) -> np.ndarray:
        """Extract variances as numpy array."""
        return np.array([result['variance'] for result in score_results])
