"""
Gradient Agent: Construct proxy gradients in language space.

Implements gradient construction with information isolation constraints,
using only aggregated statistics without access to specific sample content.
"""

from typing import List, Dict, Tuple, Callable
import numpy as np


class GradientAgent:
    """
    Constructs proxy gradients using aggregated statistics.
    
    Implements information isolation: does not access actual response content,
    only receives aggregated error statistics.
    """
    
    def __init__(self, llm_fn: Callable[[str], str], 
                 n_samples_per_category: int = 3):
        """
        Initialize gradient agent.
        
        Args:
            llm_fn: Function that takes a prompt and returns text
            n_samples_per_category: Number of samples to select from each category
        """
        self.llm_fn = llm_fn
        self.n_samples_per_category = n_samples_per_category
    
    def select_gradient_samples(self, judge_scores: np.ndarray, 
                               human_scores: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Select samples for gradient construction.
        
        Selects three categories:
        1. Overestimated: JudgeScore >> HumanScore
        2. Underestimated: JudgeScore << HumanScore
        3. Well-aligned: JudgeScore ≈ HumanScore
        
        Args:
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            
        Returns:
            Dictionary with indices for each category
        """
        errors = judge_scores - human_scores
        abs_errors = np.abs(errors)
        
        # Category 1: Overestimated (positive error)
        overestimated_idx = np.where(errors > 0)[0]
        if len(overestimated_idx) > 0:
            # Sort by error magnitude
            sorted_idx = overestimated_idx[np.argsort(-errors[overestimated_idx])]
            overestimated = sorted_idx[:self.n_samples_per_category]
        else:
            overestimated = np.array([])
        
        # Category 2: Underestimated (negative error)
        underestimated_idx = np.where(errors < 0)[0]
        if len(underestimated_idx) > 0:
            # Sort by error magnitude
            sorted_idx = underestimated_idx[np.argsort(errors[underestimated_idx])]
            underestimated = sorted_idx[:self.n_samples_per_category]
        else:
            underestimated = np.array([])
        
        # Category 3: Well-aligned (small error)
        sorted_by_abs_error = np.argsort(abs_errors)
        well_aligned = sorted_by_abs_error[:self.n_samples_per_category]
        
        return {
            'overestimated': overestimated,
            'underestimated': underestimated,
            'well_aligned': well_aligned
        }
    
    def compute_statistics(self, judge_scores: np.ndarray, 
                          human_scores: np.ndarray,
                          selected_indices: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Compute aggregated statistics (information isolation constraint).
        
        Returns only aggregated statistics, not individual sample details.
        
        Args:
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            selected_indices: Indices for each category
            
        Returns:
            Aggregated statistics dictionary
        """
        errors = judge_scores - human_scores
        
        stats = {
            'total_samples': len(judge_scores),
            'overestimated': {
                'count': len(selected_indices['overestimated']),
                'mean_error': float(np.mean(errors[selected_indices['overestimated']])) 
                             if len(selected_indices['overestimated']) > 0 else 0.0,
                'max_error': float(np.max(errors[selected_indices['overestimated']]))
                            if len(selected_indices['overestimated']) > 0 else 0.0,
            },
            'underestimated': {
                'count': len(selected_indices['underestimated']),
                'mean_error': float(np.mean(errors[selected_indices['underestimated']]))
                             if len(selected_indices['underestimated']) > 0 else 0.0,
                'min_error': float(np.min(errors[selected_indices['underestimated']]))
                            if len(selected_indices['underestimated']) > 0 else 0.0,
            },
            'well_aligned': {
                'count': len(selected_indices['well_aligned']),
                'mean_abs_error': float(np.mean(np.abs(errors[selected_indices['well_aligned']])))
                                 if len(selected_indices['well_aligned']) > 0 else 0.0,
            },
            'overall': {
                'mae': float(np.mean(np.abs(errors))),
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
            }
        }
        
        return stats
    
    def construct_proxy_gradient(self, current_prompt: str,
                                 statistics: Dict[str, any]) -> str:
        """
        Construct proxy gradient using LLM.
        
        The LLM receives only aggregated statistics and current prompt,
        not any actual sample content.
        
        Args:
            current_prompt: Current JudgePrompt text
            statistics: Aggregated statistics from compute_statistics
            
        Returns:
            Proxy gradient as structured text
        """
        gradient_prompt = f"""You are a meta-optimizer analyzing judge prompt performance.

Current Judge Prompt:
{current_prompt}

Performance Statistics:
- Total samples evaluated: {statistics['total_samples']}
- Overall MAE: {statistics['overall']['mae']:.3f}
- Mean error (bias): {statistics['overall']['mean_error']:.3f}
- Std error: {statistics['overall']['std_error']:.3f}

Error Category Analysis:
1. OVERESTIMATED samples (Judge score too high):
   - Count: {statistics['overestimated']['count']}
   - Mean error: +{statistics['overestimated']['mean_error']:.3f}
   - Max error: +{statistics['overestimated']['max_error']:.3f}

2. UNDERESTIMATED samples (Judge score too low):
   - Count: {statistics['underestimated']['count']}
   - Mean error: {statistics['underestimated']['mean_error']:.3f}
   - Min error: {statistics['underestimated']['min_error']:.3f}

3. WELL-ALIGNED samples (Judge score close to human):
   - Count: {statistics['well_aligned']['count']}
   - Mean absolute error: {statistics['well_aligned']['mean_abs_error']:.3f}

Task: Based on ONLY these aggregated statistics (without knowing specific sample content):
1. Identify abstract patterns in the prompt that might cause overestimation
2. Identify abstract patterns that might cause underestimation
3. Suggest the direction of improvement for the prompt

Output a structured analysis with:
- OVERESTIMATION_CAUSES: Abstract reasoning about why scores might be too high
- UNDERESTIMATION_CAUSES: Abstract reasoning about why scores might be too low
- IMPROVEMENT_DIRECTION: High-level guidance for prompt modification

Do NOT make assumptions about specific sample content or task domain."""

        return self.llm_fn(gradient_prompt)
    
    def compute_gradient(self, current_prompt: str,
                        judge_scores: np.ndarray,
                        human_scores: np.ndarray) -> Dict[str, any]:
        """
        Full gradient computation pipeline.
        
        Args:
            current_prompt: Current JudgePrompt text
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            
        Returns:
            Dictionary containing statistics and proxy gradient
        """
        # Select samples
        selected_indices = self.select_gradient_samples(judge_scores, human_scores)
        
        # Compute aggregated statistics
        statistics = self.compute_statistics(judge_scores, human_scores, selected_indices)
        
        # Construct proxy gradient
        proxy_gradient = self.construct_proxy_gradient(current_prompt, statistics)
        
        return {
            'statistics': statistics,
            'selected_indices': selected_indices,
            'proxy_gradient': proxy_gradient
        }
