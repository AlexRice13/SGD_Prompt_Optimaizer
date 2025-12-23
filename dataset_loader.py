"""
Dataset Loader: Load training data from JSONL files.

Supports loading datasets in JSONL format with prompt, response, and score fields.
"""

import json
from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path


class DatasetLoader:
    """
    Load and parse JSONL datasets for prompt optimization.
    
    Expected JSONL format:
    {"prompt": "...", "response": "...", "score": 8.5}
    """
    
    def __init__(self, 
                 prompt_field: str = "prompt",
                 response_field: str = "response", 
                 score_field: str = "score"):
        """
        Initialize dataset loader.
        
        Args:
            prompt_field: Key for prompt in JSONL
            response_field: Key for response in JSONL
            score_field: Key for score in JSONL
        """
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.score_field = score_field
    
    def load_jsonl(self, filepath: str) -> List[Dict]:
        """
        Load JSONL file.
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            List of dictionaries, one per line
        """
        data = []
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue
        
        return data
    
    def parse_dataset(self, data: List[Dict]) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Parse loaded data into prompts, responses, and scores.
        
        Args:
            data: List of dictionaries from JSONL
            
        Returns:
            Tuple of (prompts, responses, scores)
        """
        prompts = []
        responses = []
        scores = []
        
        for item in data:
            # Extract fields with validation
            if self.response_field not in item:
                print(f"Warning: Missing '{self.response_field}' field, skipping item")
                continue
            if self.score_field not in item:
                print(f"Warning: Missing '{self.score_field}' field, skipping item")
                continue
            
            prompt = item.get(self.prompt_field, "")
            response = item[self.response_field]
            score = float(item[self.score_field])
            
            prompts.append(prompt)
            responses.append(response)
            scores.append(score)
        
        return prompts, responses, np.array(scores)
    
    def load_dataset(self, filepath: str) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Load and parse dataset from JSONL file.
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            Tuple of (prompts, responses, scores)
        """
        data = self.load_jsonl(filepath)
        return self.parse_dataset(data)
    
    def split_dataset(self, 
                     responses: List[str],
                     scores: np.ndarray,
                     val_split: float = 0.2,
                     seed: Optional[int] = 42) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
        """
        Split dataset into train and validation sets.
        
        Args:
            responses: List of responses
            scores: Array of scores
            val_split: Fraction of data to use for validation
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_responses, train_scores, val_responses, val_scores)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = len(responses)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        n_val = int(n_samples * val_split)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_responses = [responses[i] for i in train_indices]
        train_scores = scores[train_indices]
        
        val_responses = [responses[i] for i in val_indices]
        val_scores = scores[val_indices]
        
        return train_responses, train_scores, val_responses, val_scores


if __name__ == '__main__':
    """Unit tests for DatasetLoader class."""
    import tempfile
    import os
    import numpy as np
    
    print("Running DatasetLoader unit tests...")
    
    # Test 1: Basic initialization
    print("\n1. Testing basic initialization...")
    loader = DatasetLoader()
    assert loader.prompt_field == "prompt"
    assert loader.response_field == "response"
    assert loader.score_field == "score"
    print("   ✓ Initialization works")
    
    # Test 2: Custom field names
    print("\n2. Testing custom field names...")
    loader2 = DatasetLoader(prompt_field="q", response_field="a", score_field="rating")
    assert loader2.prompt_field == "q"
    assert loader2.response_field == "a"
    assert loader2.score_field == "rating"
    print("   ✓ Custom field names work")
    
    # Test 3: Load and parse JSONL
    print("\n3. Testing load_jsonl and parse_dataset...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        filename = f.name
        f.write('{"prompt": "Q1", "response": "A1", "score": 8.5}\n')
        f.write('{"prompt": "Q2", "response": "A2", "score": 7.0}\n')
        f.write('{"prompt": "Q3", "response": "A3", "score": 9.2}\n')
    
    try:
        data = loader.load_jsonl(filename)
        assert len(data) == 3
        assert data[0]['score'] == 8.5
        
        prompts, responses, scores = loader.parse_dataset(data)
        assert len(prompts) == 3
        assert len(responses) == 3
        assert len(scores) == 3
        assert prompts[0] == "Q1"
        assert responses[1] == "A2"
        assert scores[2] == 9.2
        print("   ✓ JSONL loading and parsing work")
    finally:
        os.unlink(filename)
    
    # Test 4: load_dataset (combined)
    print("\n4. Testing load_dataset...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        filename = f.name
        f.write('{"prompt": "Test", "response": "R1", "score": 5.0}\n')
    
    try:
        prompts, responses, scores = loader.load_dataset(filename)
        assert len(responses) == 1
        assert scores[0] == 5.0
        print("   ✓ load_dataset works")
    finally:
        os.unlink(filename)
    
    # Test 5: Split dataset
    print("\n5. Testing split_dataset...")
    responses = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    
    train_resp, train_scores, val_resp, val_scores = loader.split_dataset(
        responses, scores, val_split=0.2, seed=42
    )
    
    assert len(train_resp) == 8
    assert len(val_resp) == 2
    assert len(train_scores) == 8
    assert len(val_scores) == 2
    assert len(train_resp) + len(val_resp) == len(responses)
    print(f"   Split: {len(train_resp)} train, {len(val_resp)} val ✓")
    
    # Test 6: Invalid JSON handling
    print("\n6. Testing invalid JSON handling...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        filename = f.name
        f.write('{"prompt": "Q1", "response": "A1", "score": 8.5}\n')
        f.write('invalid json line\n')
        f.write('{"prompt": "Q2", "response": "A2", "score": 7.0}\n')
    
    try:
        data = loader.load_jsonl(filename)
        assert len(data) == 2  # Invalid line skipped
        print("   ✓ Invalid JSON handled gracefully")
    finally:
        os.unlink(filename)
    
    # Test 7: Missing fields
    print("\n7. Testing missing fields...")
    data_with_missing = [
        {"prompt": "Q1", "response": "A1", "score": 8.5},
        {"prompt": "Q2", "response": "A2"},  # Missing score
        {"prompt": "Q3", "score": 7.0},     # Missing response
    ]
    prompts, responses, scores = loader.parse_dataset(data_with_missing)
    assert len(responses) == 1  # Only first item is valid
    print("   ✓ Missing fields handled gracefully")
    
    # Test 8: Empty lines in JSONL
    print("\n8. Testing empty lines...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        filename = f.name
        f.write('{"prompt": "Q1", "response": "A1", "score": 8.5}\n')
        f.write('\n')
        f.write('  \n')
        f.write('{"prompt": "Q2", "response": "A2", "score": 7.0}\n')
    
    try:
        data = loader.load_jsonl(filename)
        assert len(data) == 2  # Empty lines skipped
        print("   ✓ Empty lines handled correctly")
    finally:
        os.unlink(filename)
    
    print("\n" + "="*50)
    print("All DatasetLoader tests passed! ✓")
    print("="*50)
