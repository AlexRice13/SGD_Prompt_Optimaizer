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


if __name__ == '__main__':
    """Unit tests for constants module."""
    
    print("Running constants unit tests...")
    
    # Test 1: Constants defined
    print("\n1. Testing constants...")
    assert STRUCTURAL_EDIT_LR_THRESHOLD == 0.6
    print(f"   STRUCTURAL_EDIT_LR_THRESHOLD = {STRUCTURAL_EDIT_LR_THRESHOLD} ✓")
    
    # Test 2: can_perform_structural_edit - above threshold
    print("\n2. Testing can_perform_structural_edit - above threshold...")
    assert can_perform_structural_edit(0.7) == True
    assert can_perform_structural_edit(0.6) == True  # Exactly at threshold
    assert can_perform_structural_edit(1.0) == True
    print("   ✓ High LR allows structural edits")
    
    # Test 3: can_perform_structural_edit - below threshold
    print("\n3. Testing can_perform_structural_edit - below threshold...")
    assert can_perform_structural_edit(0.5) == False
    assert can_perform_structural_edit(0.1) == False
    assert can_perform_structural_edit(0.0) == False
    print("   ✓ Low LR blocks structural edits")
    
    # Test 4: Boundary cases
    print("\n4. Testing boundary cases...")
    assert can_perform_structural_edit(0.59999) == False
    assert can_perform_structural_edit(0.60001) == True
    print("   ✓ Boundary cases work correctly")
    
    print("\n" + "="*50)
    print("All constants tests passed! ✓")
    print("="*50)
