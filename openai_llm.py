"""
OpenAI LLM Integration: Wrapper for OpenAI API calls.

Provides LLM functions for Judge, Gradient Agent, and Optimizer using OpenAI API.
Supports concurrent API calls for efficiency.
"""

import os
import json
import re
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor


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
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text.
        Uses rough approximation: ~4 characters per token for English/Chinese.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough estimate: 4 chars per token (conservative for both English and Chinese)
        return len(text) // 4
    
    def call_llm(self, 
                 prompt: str,
                 system_message: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        """
        Call OpenAI API with prompt.
        
        Dynamic token management: Like vLLM, adjusts max_tokens based on input length
        to avoid exceeding total context limit.
        
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
        requested_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Estimate input tokens
        total_input_text = (system_message or "") + prompt
        input_tokens = self._estimate_token_count(total_input_text)
        
        # Dynamic shrink: max_tokens = requested_max_tokens - input_tokens
        # This ensures total tokens don't exceed model's context window
        # Add small buffer (100 tokens) for safety
        adjusted_max_tokens = max(100, requested_max_tokens - input_tokens - 100)
        
        # If input is very large, warn the user
        if adjusted_max_tokens < requested_max_tokens * 0.3:
            print(f"Warning: Input is very large ({input_tokens} tokens estimated). "
                  f"Max output tokens reduced from {requested_max_tokens} to {adjusted_max_tokens}.")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=adjusted_max_tokens
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise
    
    def judge_llm_fn(self, judge_prompt: str, response: str) -> float:
        """
        Judge LLM function for scoring responses.
        
        Args:
            judge_prompt: The JudgePrompt text (used as system prompt)
            response: Response to score
            
        Returns:
            Numeric score
        """
        # Use judge_prompt directly as system message
        user_prompt = f"""Response to evaluate:
{response}

Please provide your score:"""
        
        result = self.call_llm(user_prompt, system_message=judge_prompt, temperature=0.3)
        
        # Extract numeric score from response
        score = self._extract_score(result)
        return score
    
    def _extract_score(self, text: str) -> float:
        """
        Extract numeric score from LLM response using regex.
        
        Supports various formats:
        - "Score: 8.5" or "score: 8.5" or "Score:8.5" (with/without space)
        - "分数: 8.5" or "分数：8.5" (Chinese)
        - "Rating: 8.5"
        - Plain number: "8.5" or ".5" (decimal starting with point)
        
        Args:
            text: LLM response text
            
        Returns:
            Extracted score as float
        """
        # Number pattern that matches integers, decimals, and numbers starting with decimal point
        NUMBER_PATTERN = r'([0-9]*\.?[0-9]+)'
        
        # Try patterns in order of specificity
        patterns = [
            r'(?:score|Score|SCORE)[\s:：]*' + NUMBER_PATTERN,  # English "score" (optional separator)
            r'(?:分数|評分)[\s:：]*' + NUMBER_PATTERN,  # Chinese "分数" or "評分"
            r'(?:rating|Rating|RATING)[\s:：]*' + NUMBER_PATTERN,  # "rating"
            r'^' + NUMBER_PATTERN + r'$',  # Just a number on its own line
            NUMBER_PATTERN,  # Any number (last resort)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                try:
                    score = float(match.group(1))
                    return score
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
                            (already contains complete instructions from prompts.py)
            
        Returns:
            Proxy gradient as text
        """
        # gradient_prompt already contains complete Chinese instructions with terminology
        # explanations from prompts.py - no additional system message needed
        return self.call_llm(gradient_prompt, system_message=None, temperature=0.7)
    
    def optimizer_llm_fn(self, optimizer_prompt: str) -> str:
        """
        Optimizer LLM function.
        
        Args:
            optimizer_prompt: Prompt with gradient and constraints
                             (already contains complete instructions from prompts.py)
            
        Returns:
            Modification suggestion as text
        """
        # optimizer_prompt already contains complete Chinese instructions with terminology
        # explanations from prompts.py - no additional system message needed
        return self.call_llm(optimizer_prompt, system_message=None, temperature=0.5)


def create_openai_llm_functions(model: str = "gpt-4",
                                 judge_temperature: float = 0.3,
                                 gradient_temperature: float = 0.7,
                                 optimizer_temperature: float = 0.5,
                                 max_tokens: Optional[int] = None,
                                 api_key: Optional[str] = None,
                                 api_base: Optional[str] = None):
    """
    Create all three LLM functions for the trainer using OpenAI.
    
    Args:
        model: OpenAI model name
        judge_temperature: Temperature for judge LLM
        gradient_temperature: Temperature for gradient LLM
        optimizer_temperature: Temperature for optimizer LLM
        max_tokens: Maximum output tokens (if None, uses config from trainer; dynamically adjusted based on input)
        api_key: API key (if None, reads from env)
        api_base: API base URL (if None, reads from env)
        
    Returns:
        Tuple of (judge_fn, gradient_fn, optimizer_fn)
    """
    # Use provided max_tokens or fall back to OpenAILLM's default (2000)
    # Typically max_tokens should be passed from trainer config
    actual_max_tokens = max_tokens if max_tokens is not None else 2000
    
    # Create instances with appropriate temperatures and max_tokens for each role
    judge_llm = OpenAILLM(model=model, temperature=judge_temperature, max_tokens=actual_max_tokens, 
                         api_key=api_key, api_base=api_base)
    gradient_llm = OpenAILLM(model=model, temperature=gradient_temperature, max_tokens=actual_max_tokens,
                            api_key=api_key, api_base=api_base)
    optimizer_llm = OpenAILLM(model=model, temperature=optimizer_temperature, max_tokens=actual_max_tokens,
                             api_key=api_key, api_base=api_base)
    
    def judge_fn(prompt: str, response: str) -> float:
        return judge_llm.judge_llm_fn(prompt, response)
    
    def gradient_fn(prompt: str) -> str:
        return gradient_llm.gradient_llm_fn(prompt)
    
    def optimizer_fn(prompt: str) -> str:
        return optimizer_llm.optimizer_llm_fn(prompt)
    
    return judge_fn, gradient_fn, optimizer_fn
