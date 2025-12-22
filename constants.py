"""
Constants used across the SGD Prompt Optimization Framework.
"""

# Learning rate threshold for structural edits (add/remove sections)
# When LR >= STRUCTURAL_EDIT_LR_THRESHOLD: can add/remove sections
# When LR < STRUCTURAL_EDIT_LR_THRESHOLD: can only edit existing sections
STRUCTURAL_EDIT_LR_THRESHOLD = 0.6
