"""
Example usage of the SGD Prompt Optimization Framework.

This script demonstrates how to use the framework with mock LLM functions.
"""

import numpy as np
from judge_prompt import JudgePrompt
from trainer import SGDPromptTrainer


def mock_judge_llm(prompt: str, response: str) -> float:
    """
    Mock Judge LLM function.
    
    In practice, this would call a real LLM API with the prompt and response,
    and parse the returned score.
    """
    # Simulate scoring based on response length and simple heuristics
    score = min(10.0, max(1.0, len(response) / 20 + np.random.normal(5, 1)))
    return score


def mock_gradient_llm(prompt: str) -> str:
    """
    Mock Gradient Agent LLM function.
    
    In practice, this would call a strong LLM to analyze statistics
    and generate proxy gradient.
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
    
    In practice, this would call a strong LLM to generate
    specific modification suggestions.
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


def create_example_prompt() -> JudgePrompt:
    """Create an example initial prompt."""
    sections = {
        "Scoring Criteria": "Evaluate responses based on completeness and clarity.",
        "Scale": "Use a scale from 1 to 10, where 1 is poor and 10 is excellent.",
        "Output Format": "Output only the numeric score, nothing else."
    }
    
    editable_sections = ["Scoring Criteria"]
    
    return JudgePrompt(sections, editable_sections)


def generate_mock_data(n_samples: int = 100):
    """Generate mock training/validation data."""
    np.random.seed(42)
    
    # Generate mock responses with varying quality
    responses = []
    human_scores = []
    
    for i in range(n_samples):
        # Simulate responses of different quality
        quality = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])
        
        if quality == 'low':
            response = "Short response. " * np.random.randint(1, 3)
            score = np.random.uniform(1, 4)
        elif quality == 'medium':
            response = "This is a medium quality response with some detail. " * np.random.randint(2, 5)
            score = np.random.uniform(4, 7)
        else:
            response = "This is a high quality, comprehensive response with excellent detail and accuracy. " * np.random.randint(3, 6)
            score = np.random.uniform(7, 10)
        
        responses.append(response)
        human_scores.append(score)
    
    return responses, np.array(human_scores)


def main():
    """Main example execution."""
    print("SGD Prompt Optimization Framework - Example Usage")
    print("=" * 60)
    
    # Create initial prompt
    print("\n1. Creating initial JudgePrompt...")
    initial_prompt = create_example_prompt()
    print(f"Prompt has {len(initial_prompt.sections)} sections")
    print(f"Editable sections: {', '.join(initial_prompt.editable_sections)}")
    
    # Generate mock data
    print("\n2. Generating mock training/validation data...")
    train_responses, train_scores = generate_mock_data(100)
    val_responses, val_scores = generate_mock_data(30)
    print(f"Train: {len(train_responses)} samples")
    print(f"Val: {len(val_responses)} samples")
    
    # Configure trainer
    print("\n3. Configuring trainer...")
    config = {
        'max_steps': 10,  # Small number for demo
        'batch_size': 20,
        'initial_lr': 0.1,
        'min_lr': 0.01,
        'warmup_steps': 2,
        'patience': 5,
        'enable_version_control': False,  # Disable for demo
    }
    
    # Create trainer
    print("\n4. Initializing trainer...")
    trainer = SGDPromptTrainer(
        judge_llm_fn=mock_judge_llm,
        gradient_llm_fn=mock_gradient_llm,
        optimizer_llm_fn=mock_optimizer_llm,
        initial_prompt=initial_prompt,
        train_responses=train_responses,
        train_human_scores=train_scores,
        val_responses=val_responses,
        val_human_scores=val_scores,
        config=config
    )
    
    # Train
    print("\n5. Starting training...")
    print("=" * 60)
    best_prompt = trainer.train()
    
    # Results
    print("\n" + "=" * 60)
    print("6. Training completed!")
    print("\nBest Prompt:")
    print(best_prompt.get_full_prompt())
    
    # Save history
    print("\n7. Saving training history...")
    trainer.save_history("training_history.json")
    print("History saved to training_history.json")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
