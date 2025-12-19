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
