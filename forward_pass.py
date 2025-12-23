"""
Forward Pass: Score responses using JudgePrompt.

Implements the forward pass where JudgePrompt is used to score responses,
with support for self-consistency variance estimation and concurrent LLM calls.
"""

from typing import List, Dict, Callable, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor


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


if __name__ == '__main__':
    """Unit tests for ForwardPass class."""
    import numpy as np
    
    print("Running ForwardPass unit tests...")
    
    # Test 1: Basic initialization
    print("\n1. Testing basic initialization...")
    
    def mock_judge_fn(prompt, response):
        """Mock judge function that returns a deterministic score."""
        return 5.0 + len(response) * 0.1
    
    forward_pass = ForwardPass(mock_judge_fn, n_consistency_samples=3)
    assert forward_pass.n_consistency_samples == 3
    print("   ✓ Initialization works")
    
    # Test 2: Score single response
    print("\n2. Testing score_response...")
    prompt = "Test prompt"
    response = "Test response"
    result = forward_pass.score_response(prompt, response)
    
    assert 'mean_score' in result
    assert 'variance' in result
    assert 'all_scores' in result
    assert len(result['all_scores']) == 3
    print(f"   Mean: {result['mean_score']:.2f}, Variance: {result['variance']:.4f} ✓")
    
    # Test 3: Score batch
    print("\n3. Testing score_batch...")
    responses = ["Response 1", "Response 2", "Response 3"]
    results = forward_pass.score_batch(prompt, responses)
    
    assert len(results) == 3
    assert all('mean_score' in r for r in results)
    print(f"   Scored {len(results)} responses ✓")
    
    # Test 4: Get scores array
    print("\n4. Testing get_scores_array...")
    scores = forward_pass.get_scores_array(results)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 3
    print(f"   Scores: {scores} ✓")
    
    # Test 5: Get variances array
    print("\n5. Testing get_variances_array...")
    variances = forward_pass.get_variances_array(results)
    assert isinstance(variances, np.ndarray)
    assert len(variances) == 3
    assert all(v >= 0 for v in variances)
    print(f"   Variances: {variances} ✓")
    
    # Test 6: Single consistency sample
    print("\n6. Testing n_consistency_samples=1...")
    fp_single = ForwardPass(mock_judge_fn, n_consistency_samples=1)
    result_single = fp_single.score_response(prompt, response)
    assert result_single['variance'] == 0.0  # No variance with single sample
    print("   ✓ Single sample works (variance=0)")
    
    print("\n" + "="*50)
    print("All ForwardPass tests passed! ✓")
    print("="*50)
