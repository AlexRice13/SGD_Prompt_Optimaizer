"""
Constants and utility functions used across the SGD Prompt Optimization Framework.
"""

# Learning rate threshold for structural edits (add/remove sections)
# When LR >= STRUCTURAL_EDIT_LR_THRESHOLD: can add/remove sections
# When LR < STRUCTURAL_EDIT_LR_THRESHOLD: can only edit existing sections
STRUCTURAL_EDIT_LR_THRESHOLD = 0.6


def can_perform_structural_edit(learning_rate: float) -> bool:
    """
    Check if structural edits (add/remove sections) are allowed at current learning rate.
    
    Args:
        learning_rate: Current learning rate
        
    Returns:
        True if structural edits are allowed (LR >= threshold), False otherwise
    """
    return learning_rate >= STRUCTURAL_EDIT_LR_THRESHOLD
