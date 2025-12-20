"""
Example usage of the SGD Prompt Optimization Framework.

This script demonstrates how to use the framework with real OpenAI API
and dataset loading from JSONL files.
"""

import os
import numpy as np
from pathlib import Path
from judge_prompt import JudgePrompt
from trainer import SGDPromptTrainer
from dataset_loader import DatasetLoader
from openai_llm import create_openai_llm_functions


def check_environment():
    """Check if required environment variables are set."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        print("\nFor testing without OpenAI API, you can use the mock functions.")
        return False
    return True


def load_dataset_from_file(filepath: str, val_split: float = 0.2):
    """
    Load dataset from JSONL file.
    
    Args:
        filepath: Path to JSONL file
        val_split: Fraction for validation split
        
    Returns:
        Tuple of (train_responses, train_scores, val_responses, val_scores)
    """
    print(f"Loading dataset from {filepath}...")
    
    loader = DatasetLoader(
        prompt_field="prompt",
        response_field="response",
        score_field="score"
    )
    
    # Load and parse dataset
    prompts, responses, scores = loader.load_dataset(filepath)
    print(f"Loaded {len(responses)} samples")
    
    # Split into train/val
    train_responses, train_scores, val_responses, val_scores = loader.split_dataset(
        responses, scores, val_split=val_split, seed=42
    )
    
    print(f"Train: {len(train_responses)} samples, Val: {len(val_responses)} samples")
    
    return train_responses, train_scores, val_responses, val_scores


def create_sample_dataset(output_path: str = "sample_dataset.jsonl", n_samples: int = 50):
    """
    Create a sample JSONL dataset for demonstration.
    
    Args:
        output_path: Path to output JSONL file
        n_samples: Number of samples to generate
    """
    import json
    
    print(f"Creating sample dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(n_samples):
            # Generate sample data with varying quality
            quality = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])
            
            if quality == 'low':
                response = "Short response. " * np.random.randint(1, 3)
                score = float(np.random.uniform(1, 4))
            elif quality == 'medium':
                response = "This is a medium quality response with some detail. " * np.random.randint(2, 5)
                score = float(np.random.uniform(4, 7))
            else:
                response = "This is a high quality, comprehensive response with excellent detail and accuracy. " * np.random.randint(3, 6)
                score = float(np.random.uniform(7, 10))
            
            item = {
                "prompt": "Evaluate this response",
                "response": response,
                "score": round(score, 2)
            }
            
            f.write(json.dumps(item) + '\n')
    
    print(f"Sample dataset created at {output_path}")
    return output_path


def create_example_prompt() -> JudgePrompt:
    """Create an example initial prompt."""
    sections = {
        "Scoring Criteria": "Evaluate responses based on completeness and clarity.",
        "Scale": "Use a scale from 1 to 10, where 1 is poor and 10 is excellent.",
        "Output Format": "Output only the numeric score, nothing else."
    }
    
    # Scale and Output Format are meta sections (frozen, cannot be modified)
    # Scoring Criteria is editable
    meta_sections = ["Scale", "Output Format"]
    
    return JudgePrompt(sections, meta_sections)


def create_example_prompt_file(output_path: str = "initial_judge_prompt.json") -> str:
    """
    Create an example JudgePrompt JSON file for demonstration.
    
    The JSON file format is:
    {
        "sections": {
            "Scoring Criteria": "...",
            "Scale": "...",
            "Output Format": "..."
        },
        "meta_sections": ["Scale", "Output Format"]
    }
    
    Args:
        output_path: Path to output JSON file
        
    Returns:
        Path to created file
    """
    print(f"Creating example JudgePrompt file at {output_path}...")
    
    prompt = create_example_prompt()
    prompt.save(output_path)
    
    print(f"Example JudgePrompt file created at {output_path}")
    print(f"File format:")
    with open(output_path, 'r', encoding='utf-8') as f:
        print(f.read())
    
    return output_path


def load_prompt_from_file(filepath: str) -> JudgePrompt:
    """
    Load JudgePrompt from JSON file.
    
    Args:
        filepath: Path to JSON file containing JudgePrompt
        
    Returns:
        Loaded JudgePrompt instance
    """
    print(f"Loading JudgePrompt from {filepath}...")
    prompt = JudgePrompt.load(filepath)
    print(f"Loaded prompt with {len(prompt.sections)} sections")
    print(f"Meta sections (frozen): {', '.join(prompt.meta_sections)}")
    print(f"Editable sections: {', '.join(prompt.get_editable_sections())}")
    return prompt


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
    print("SGD Prompt Optimization Framework - Example Usage with OpenAI")
    print("=" * 70)
    
    # Check environment
    use_openai = check_environment()
    
    # Create or load initial prompt
    print("\n1. Loading/Creating initial JudgePrompt...")
    prompt_path = os.environ.get("PROMPT_PATH", "initial_judge_prompt.json")
    
    if Path(prompt_path).exists():
        # Load existing prompt from JSON file
        initial_prompt = load_prompt_from_file(prompt_path)
    else:
        # Create example prompt file if it doesn't exist
        print(f"Prompt file not found at {prompt_path}")
        create_example_prompt_file(prompt_path)
        initial_prompt = JudgePrompt.load(prompt_path)
        print(f"Created and loaded example prompt")
    
    print(f"Prompt has {len(initial_prompt.sections)} sections")
    print(f"Meta sections (frozen): {', '.join(initial_prompt.meta_sections)}")
    print(f"Editable sections: {', '.join(initial_prompt.get_editable_sections())}")
    
    # Load or create dataset
    print("\n2. Loading dataset...")
    dataset_path = os.environ.get("DATASET_PATH", "sample_dataset.jsonl")
    
    # Create sample dataset if it doesn't exist
    if not Path(dataset_path).exists():
        print(f"Dataset not found at {dataset_path}")
        dataset_path = create_sample_dataset(dataset_path, n_samples=50)
    
    # Load dataset from JSONL file
    train_responses, train_scores, val_responses, val_scores = load_dataset_from_file(
        dataset_path, val_split=0.2
    )
    
    # Setup LLM functions
    print("\n3. Setting up LLM functions...")
    if use_openai:
        print("Using OpenAI API")
        model = os.environ.get("OPENAI_MODEL", "gpt-4")
        print(f"Model: {model}")
        
        try:
            judge_fn, gradient_fn, optimizer_fn = create_openai_llm_functions(
                model=model,
                judge_temperature=0.3,
                gradient_temperature=0.7,
                optimizer_temperature=0.5
            )
            print("OpenAI functions created successfully")
        except Exception as e:
            print(f"Error creating OpenAI functions: {e}")
            print("Falling back to mock functions")
            use_openai = False
    
    if not use_openai:
        print("Using mock LLM functions (for testing without API)")
        from example_mock_functions import create_mock_llm_functions
        judge_fn, gradient_fn, optimizer_fn = create_mock_llm_functions()
    
    # Configure trainer
    print("\n4. Configuring trainer...")
    config = {
        'max_steps': int(os.environ.get("MAX_STEPS", "10")),
        'batch_size': int(os.environ.get("BATCH_SIZE", "16")),
        'initial_lr': float(os.environ.get("INITIAL_LR", "0.1")),
        'min_lr': float(os.environ.get("MIN_LR", "0.01")),
        'warmup_steps': int(os.environ.get("WARMUP_STEPS", "2")),
        'patience': int(os.environ.get("PATIENCE", "5")),
        'enable_version_control': os.environ.get("ENABLE_VERSION_CONTROL", "false").lower() == "true",
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create trainer
    print("\n5. Initializing trainer...")
    trainer = SGDPromptTrainer(
        judge_llm_fn=judge_fn,
        gradient_llm_fn=gradient_fn,
        optimizer_llm_fn=optimizer_fn,
        initial_prompt=initial_prompt,
        train_responses=train_responses,
        train_human_scores=train_scores,
        val_responses=val_responses,
        val_human_scores=val_scores,
        config=config
    )
    
    # Train
    print("\n6. Starting training...")
    print("=" * 70)
    best_prompt = trainer.train()
    
    # Results
    print("\n" + "=" * 70)
    print("7. Training completed!")
    print("\nBest Prompt:")
    print(best_prompt.get_full_prompt())
    
    # Save results
    print("\n8. Saving results...")
    best_prompt.save("best_prompt.json")
    print("Best prompt saved to best_prompt.json")
    
    trainer.save_history("training_history.json")
    print("Training history saved to training_history.json")
    
    print("\nExample completed successfully!")
    print("\nTo use with your own data:")
    print("  1. Create a JudgePrompt JSON file (initial_judge_prompt.json):")
    print("     {")
    print("       \"sections\": {")
    print("         \"Scoring Criteria\": \"Your criteria...\",")
    print("         \"Scale\": \"Use 1-10 scale...\",")
    print("         \"Output Format\": \"Output only the numeric score.\"")
    print("       },")
    print("       \"meta_sections\": [\"Scale\", \"Output Format\"]")
    print("     }")
    print("     Note: meta_sections are frozen and cannot be modified.")
    print("     All other sections are automatically editable.")
    print("  2. Create a JSONL dataset file with format:")
    print("     {\"prompt\": \"...\", \"response\": \"...\", \"score\": 8.5}")
    print("  3. Set environment variables:")
    print("     export PROMPT_PATH=/path/to/your/initial_judge_prompt.json")
    print("     export DATASET_PATH=/path/to/your/dataset.jsonl")
    print("     export OPENAI_API_KEY=your-api-key")
    print("  4. Run: python example_usage.py")


if __name__ == "__main__":
    main()
