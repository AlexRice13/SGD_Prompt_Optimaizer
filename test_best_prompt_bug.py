"""
Test to reproduce the best_prompt return bug.

This test simulates a scenario where:
1. Initial evaluation gives a certain validation loss
2. Training improves the validation loss at step N
3. Early stopping triggers
4. The returned best_prompt should be from step N, not the initial prompt
"""

import numpy as np
from judge_prompt import JudgePrompt
from trainer import SGDPromptTrainer


def create_test_prompt():
    """Create a test judge prompt."""
    sections = {
        "Scoring Criteria": "Evaluate responses based on completeness and clarity.",
        "Scale": "Use a scale from 1 to 10, where 1 is poor and 10 is excellent.",
        "Output Format": "Output only the numeric score, nothing else."
    }
    meta_sections = ["Scale", "Output Format"]
    return JudgePrompt(sections, meta_sections)


def mock_judge_fn(prompt_text, response):
    """Mock judge function that gives WORSE scores with modifications."""
    # If the prompt contains "very carefully", give WORSE alignment with human scores
    if "very carefully" in prompt_text:
        # Worse alignment - the modification makes things worse!
        base_score = len(response.split()) * 0.05
    else:
        # Better alignment with initial prompt
        base_score = len(response.split()) * 0.15
    return min(10.0, max(1.0, base_score + np.random.uniform(-0.5, 0.5)))


def mock_gradient_fn(prompt_text):
    """Mock gradient function that suggests adding 'very carefully'."""
    return """
Based on the analysis, the prompt should be more specific.
The gradient suggests:
- Section to optimize: Scoring Criteria
- Direction: Add emphasis on careful evaluation
- Simple gradient: Add "very carefully" to the criteria

SIMPLE_GRADIENT:
{
    "section_to_opti": "Scoring Criteria",
    "opti_direction": "Add emphasis on careful evaluation to reduce variance"
}
"""


def mock_optimizer_fn(prompt_text):
    """Mock optimizer function that adds 'very carefully' to the criteria."""
    # The parser expects just the new content, not JSON
    return "Evaluate responses very carefully based on completeness and clarity."


def main():
    """Run the test."""
    print("=" * 80)
    print("Testing best_prompt return bug")
    print("=" * 80)
    
    # Create test data
    np.random.seed(42)
    n_train = 40
    n_val = 10
    
    # Generate responses with varying lengths
    train_responses = [f"Response with {i} words. " * (i % 5 + 1) for i in range(n_train)]
    val_responses = [f"Response with {i} words. " * (i % 5 + 1) for i in range(n_val)]
    
    # Human scores roughly based on length
    train_scores = np.array([min(10.0, (i % 10) + np.random.uniform(0, 3)) for i in range(n_train)])
    val_scores = np.array([min(10.0, (i % 10) + np.random.uniform(0, 3)) for i in range(n_val)])
    
    # Create initial prompt
    initial_prompt = create_test_prompt()
    # Keep a copy of the initial state for comparison
    initial_criteria = initial_prompt.sections['Scoring Criteria']
    print("\nInitial prompt Scoring Criteria:")
    print(f"  '{initial_criteria}'")
    
    # Configure trainer for quick test
    config = {
        'max_steps': 5,
        'batch_size': 20,
        'initial_lr': 0.8,  # High LR to allow modifications
        'patience': 3,
        'logging_steps': 1,
        'eval_steps': 1,
        'enable_version_control': False,
        'debug': True,
    }
    
    # Create trainer
    trainer = SGDPromptTrainer(
        judge_llm_fn=mock_judge_fn,
        gradient_llm_fn=mock_gradient_fn,
        optimizer_llm_fn=mock_optimizer_fn,
        initial_prompt=initial_prompt,
        train_responses=train_responses,
        train_human_scores=train_scores,
        val_responses=val_responses,
        val_human_scores=val_scores,
        config=config
    )
    
    # Run training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    best_prompt = trainer.train()
    
    # Check results
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)
    
    print(f"\nBest step reported: {trainer.early_stopping.get_best_step()}")
    print(f"Best val loss reported: {trainer.early_stopping.get_best_loss():.4f}")
    
    print("\nInitial prompt Scoring Criteria (saved before training):")
    print(f"  '{initial_criteria}'")
    
    print("\nReturned best_prompt Scoring Criteria:")
    print(f"  '{best_prompt.sections['Scoring Criteria']}'")
    
    print("\nCurrent trainer prompt Scoring Criteria:")
    print(f"  '{trainer.current_prompt.sections['Scoring Criteria']}'")
    
    # Check if best_prompt is different from initial
    is_different = (best_prompt.sections['Scoring Criteria'] != initial_criteria)
    
    print("\n" + "=" * 80)
    if is_different:
        print("✓ TEST PASSED: best_prompt is different from initial_prompt")
    else:
        print("✗ TEST FAILED: best_prompt is the same as initial_prompt")
        print("  This is the bug! The trainer should return the improved prompt.")
    print("=" * 80)
    
    # Additional check: compare with early_stopping best step
    if trainer.early_stopping.get_best_step() > 0:
        print(f"\nNote: Early stopping says best step is {trainer.early_stopping.get_best_step()}")
        print("The returned best_prompt should be from that step, not step 0.")


if __name__ == "__main__":
    main()
