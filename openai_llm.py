"""
OpenAI LLM Integration: Wrapper for OpenAI API calls.

Provides LLM functions for Judge, Gradient Agent, and Optimizer using OpenAI API.
"""

import os
import json
import re
from typing import Optional, Dict, Any
import time


class OpenAILLM:
    """
    OpenAI LLM wrapper for the optimization framework.
    
    Gets API key and endpoint from environment variables:
    - OPENAI_API_KEY: Your OpenAI API key
    - OPENAI_API_BASE: Optional custom API endpoint (defaults to OpenAI)
    """
    
    def __init__(self,
                 model: str = "gpt-4",
                 temperature: float = 0.7,
                 max_tokens: int = 2000,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None):
        """
        Initialize OpenAI LLM client.
        
        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: API key (if None, reads from OPENAI_API_KEY env var)
            api_base: API base URL (if None, reads from OPENAI_API_BASE env var)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Get API key from env or parameter
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Get API base from env or parameter (optional)
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE")
        
        # Import OpenAI library
        try:
            import openai
            self.openai = openai
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        except ImportError:
            raise ImportError(
                "openai package not found. Install it with: pip install openai"
            )
    
    def call_llm(self, 
                 prompt: str,
                 system_message: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        """
        Call OpenAI API with prompt.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            LLM response text
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise
    
    def judge_llm_fn(self, judge_prompt: str, response: str) -> float:
        """
        Judge LLM function for scoring responses.
        
        Args:
            judge_prompt: The JudgePrompt text
            response: Response to score
            
        Returns:
            Numeric score
        """
        full_prompt = f"""{judge_prompt}

Response to evaluate:
{response}

Please provide your score:"""
        
        system_msg = "You are a fair and consistent judge. Follow the scoring criteria exactly."
        
        result = self.call_llm(full_prompt, system_message=system_msg, temperature=0.3)
        
        # Extract numeric score from response
        score = self._extract_score(result)
        return score
    
    def _extract_score(self, text: str) -> float:
        """
        Extract numeric score from LLM response.
        
        Args:
            text: LLM response text
            
        Returns:
            Extracted score as float
        """
        # Try to find a number in the text
        # Look for patterns like "8.5", "8", "Score: 7.5", etc.
        patterns = [
            r'(?:score|rating)[\s:]*([0-9]+\.?[0-9]*)',  # "score: 8.5"
            r'^([0-9]+\.?[0-9]*)',  # Number at start
            r'([0-9]+\.?[0-9]*)',  # Any number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        # Default fallback
        print(f"Warning: Could not extract score from: {text[:100]}")
        return 5.0
    
    def gradient_llm_fn(self, gradient_prompt: str) -> str:
        """
        Gradient Agent LLM function.
        
        Args:
            gradient_prompt: Prompt with statistics for gradient construction
            
        Returns:
            Proxy gradient as text
        """
        system_msg = (
            "You are a meta-optimizer analyzing judge prompt performance. "
            "Provide structured analysis based only on the statistics provided."
        )
        
        return self.call_llm(gradient_prompt, system_message=system_msg, temperature=0.7)
    
    def optimizer_llm_fn(self, optimizer_prompt: str) -> str:
        """
        Optimizer LLM function.
        
        Args:
            optimizer_prompt: Prompt with gradient and constraints
            
        Returns:
            Modification suggestion as text
        """
        system_msg = (
            "You are a prompt optimizer. Generate precise, minimal modifications "
            "that address the identified issues while respecting all constraints."
        )
        
        return self.call_llm(optimizer_prompt, system_message=system_msg, temperature=0.5)


def create_openai_llm_functions(model: str = "gpt-4",
                                 judge_temperature: float = 0.3,
                                 gradient_temperature: float = 0.7,
                                 optimizer_temperature: float = 0.5,
                                 api_key: Optional[str] = None,
                                 api_base: Optional[str] = None):
    """
    Create all three LLM functions for the trainer using OpenAI.
    
    Args:
        model: OpenAI model name
        judge_temperature: Temperature for judge LLM
        gradient_temperature: Temperature for gradient LLM
        optimizer_temperature: Temperature for optimizer LLM
        api_key: API key (if None, reads from env)
        api_base: API base URL (if None, reads from env)
        
    Returns:
        Tuple of (judge_fn, gradient_fn, optimizer_fn)
    """
    llm = OpenAILLM(model=model, api_key=api_key, api_base=api_base)
    
    def judge_fn(prompt: str, response: str) -> float:
        return llm.judge_llm_fn(prompt, response)
    
    def gradient_fn(prompt: str) -> str:
        return llm.gradient_llm_fn(prompt)
    
    def optimizer_fn(prompt: str) -> str:
        return llm.optimizer_llm_fn(prompt)
    
    return judge_fn, gradient_fn, optimizer_fn
