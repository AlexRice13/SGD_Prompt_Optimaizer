# Usage Guide: Dataset Loader and OpenAI Integration

This guide shows how to use the new dataset loader and OpenAI API integration features.

## Quick Start with OpenAI API

### 1. Install Dependencies

```bash
pip install numpy scipy openai
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY='sk-your-api-key-here'
export OPENAI_MODEL='gpt-4'  # Optional, defaults to gpt-4
```

### 3. Prepare Your Dataset

Create a JSONL file where each line is a JSON object with these fields:
- `prompt`: The original prompt (optional, can be empty string)
- `response`: The response to evaluate
- `score`: Human-provided score (numeric)

Example `my_dataset.jsonl`:
```jsonl
{"prompt": "", "response": "This is a great answer with detailed information.", "score": 9.0}
{"prompt": "", "response": "This is okay.", "score": 5.5}
{"prompt": "", "response": "Poor quality response.", "score": 2.0}
```

### 4. Run the Example

```bash
export DATASET_PATH='my_dataset.jsonl'
python example_usage.py
```

## Detailed Usage

### Loading Datasets

```python
from dataset_loader import DatasetLoader

# Initialize loader
loader = DatasetLoader(
    prompt_field="prompt",      # JSON key for prompt
    response_field="response",  # JSON key for response
    score_field="score"         # JSON key for score
)

# Load dataset
prompts, responses, scores = loader.load_dataset("dataset.jsonl")

# Split into train/validation
train_resp, train_scores, val_resp, val_scores = loader.split_dataset(
    responses, scores, 
    val_split=0.2,  # 20% for validation
    seed=42         # For reproducibility
)
```

### Using OpenAI API

```python
from openai_llm import create_openai_llm_functions

# Create all three LLM functions
judge_fn, gradient_fn, optimizer_fn = create_openai_llm_functions(
    model="gpt-4",                    # Model to use
    judge_temperature=0.3,            # Low temp for consistent scoring
    gradient_temperature=0.7,         # Higher temp for creative analysis
    optimizer_temperature=0.5,        # Medium temp for suggestions
    api_key=None,                     # Uses OPENAI_API_KEY env var
    api_base=None                     # Uses OPENAI_API_BASE env var
)

# Use in trainer
from trainer import SGDPromptTrainer

trainer = SGDPromptTrainer(
    judge_llm_fn=judge_fn,
    gradient_llm_fn=gradient_fn,
    optimizer_llm_fn=optimizer_fn,
    # ... other parameters
)
```

### Custom LLM Functions

If you want to use a different LLM provider, implement these three functions:

```python
def my_judge_llm(judge_prompt: str, response: str) -> float:
    """
    Score a response using the judge prompt.
    Returns: numeric score (e.g., 1-10)
    """
    # Your implementation here
    return score

def my_gradient_llm(gradient_prompt: str) -> str:
    """
    Analyze statistics and generate proxy gradient.
    Returns: structured text analysis
    """
    # Your implementation here
    return analysis_text

def my_optimizer_llm(optimizer_prompt: str) -> str:
    """
    Generate modification suggestion.
    Returns: structured modification in specific format
    """
    # Your implementation here
    return modification_text
```

## Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | - | Yes (for OpenAI) |
| `OPENAI_API_BASE` | Custom API endpoint | OpenAI default | No |
| `OPENAI_MODEL` | Model name | `gpt-4` | No |
| `DATASET_PATH` | Path to dataset file | `sample_dataset.jsonl` | No |
| `MAX_STEPS` | Training iterations | `10` | No |
| `BATCH_SIZE` | Batch size | `16` | No |
| `INITIAL_LR` | Initial learning rate | `0.1` | No |
| `MIN_LR` | Minimum learning rate | `0.01` | No |
| `WARMUP_STEPS` | Warmup steps | `2` | No |
| `PATIENCE` | Early stopping patience | `5` | No |
| `ENABLE_VERSION_CONTROL` | Enable git versioning | `false` | No |

## Complete Example

```python
import os
from judge_prompt import JudgePrompt
from trainer import SGDPromptTrainer
from dataset_loader import DatasetLoader
from openai_llm import create_openai_llm_functions

# 1. Set API key
os.environ['OPENAI_API_KEY'] = 'your-key-here'

# 2. Define initial prompt
initial_prompt = JudgePrompt(
    sections={
        "Scoring Criteria": "Evaluate responses based on completeness and clarity.",
        "Scale": "Use a scale from 1 to 10.",
        "Output Format": "Output only the numeric score."
    },
    meta_sections=["Scale", "Output Format"]  # These cannot be modified
)

# 3. Load dataset
loader = DatasetLoader()
_, responses, scores = loader.load_dataset("dataset.jsonl")
train_resp, train_scores, val_resp, val_scores = loader.split_dataset(
    responses, scores, val_split=0.2
)

# 4. Create LLM functions
judge_fn, gradient_fn, optimizer_fn = create_openai_llm_functions(model="gpt-4")

# 5. Configure training
config = {
    'max_steps': 50,
    'batch_size': 32,
    'initial_lr': 0.1,
    'min_lr': 0.001,
    'warmup_steps': 5,
    'alpha': 1.0,
    'beta': 1.0,
    'patience': 10,
}

# 6. Train
trainer = SGDPromptTrainer(
    judge_llm_fn=judge_fn,
    gradient_llm_fn=gradient_fn,
    optimizer_llm_fn=optimizer_fn,
    initial_prompt=initial_prompt,
    train_responses=train_resp,
    train_human_scores=train_scores,
    val_responses=val_resp,
    val_human_scores=val_scores,
    config=config
)

best_prompt = trainer.train()

# 7. Save results
best_prompt.save("optimized_prompt.json")
trainer.save_history("training_history.json")
```

## Troubleshooting

### Error: OpenAI API key not found
- Make sure you've set `OPENAI_API_KEY` environment variable
- Or pass `api_key` parameter to `create_openai_llm_functions()`

### Error: openai package not found
- Install it: `pip install openai`

### Dataset loading issues
- Check JSONL format: each line must be valid JSON
- Verify required fields: `response` and `score` are mandatory
- Check file encoding: should be UTF-8

### Example runs but uses mock functions
- This happens when `OPENAI_API_KEY` is not set
- Mock functions are used for testing without API access
- Set the API key to use real OpenAI models

## Testing Without API

To test the framework without OpenAI API:

```bash
# Just run the example without setting API key
python example_usage.py
```

The example will automatically fall back to mock functions and create a sample dataset.
