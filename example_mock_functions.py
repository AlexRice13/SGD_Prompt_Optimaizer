"""
Mock LLM functions for testing without OpenAI API.

These functions simulate LLM behavior for demonstration and testing purposes.
"""

import numpy as np


def create_mock_llm_functions():
    """
    Create mock LLM functions for testing.
    
    Returns:
        Tuple of (judge_fn, gradient_fn, optimizer_fn)
    """
    
    def mock_judge_llm(prompt: str, response: str) -> float:
        """
        Mock Judge LLM function.
        
        Simulates scoring based on response length and simple heuristics.
        """
        score = min(10.0, max(1.0, len(response) / 20 + np.random.normal(5, 1)))
        return score
    
    def mock_gradient_llm(prompt: str) -> str:
        """
        Mock Gradient Agent LLM function.
        
        Returns a simple template gradient analysis.
        """
        return """
OVERESTIMATION_CAUSES:
- The prompt may be over-rewarding verbose responses
- Lack of quality assessment beyond length

UNDERESTIMATION_CAUSES:
- The prompt may not capture concise but high-quality responses
- Missing criteria for content accuracy

IMPROVEMENT_DIRECTION:
- Add explicit quality metrics beyond length
- Balance between conciseness and completeness
- Include accuracy/correctness criteria
"""
    
    def mock_optimizer_llm(prompt: str) -> str:
        """
        Mock Optimizer LLM function.
        
        Returns a simple template modification suggestion.
        """
        return """
SECTION_TO_MODIFY: Scoring Criteria
OLD_CONTENT:
Evaluate responses based on completeness and clarity.
NEW_CONTENT:
Evaluate responses based on completeness, clarity, and accuracy. 
Balance conciseness with thoroughness - prefer responses that are 
comprehensive yet concise over those that are merely lengthy.
RATIONALE: Added accuracy criterion and guidance to avoid over-rewarding verbosity
"""
    
    return mock_judge_llm, mock_gradient_llm, mock_optimizer_llm
