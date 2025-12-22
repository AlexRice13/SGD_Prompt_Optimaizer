"""
Response Parser Utilities: Handle reasoning models and extract content outside think tags.

This module provides utilities for:
1. Extracting reasoning content from reasoning models (e.g., o1, o3)
2. Removing <think></think> tags and extracting content outside them
3. Providing safe fallbacks when responses are None or empty
"""

import re
import warnings
from typing import Optional, Tuple


def extract_content_from_response(response_obj) -> Tuple[str, Optional[str]]:
    """
    Extract content from OpenAI response object, handling reasoning models.
    
    For reasoning models (e.g., o1, o3), the response may have:
    - message.content: The final answer
    - message.reasoning_content: The reasoning process (inside <think> tags)
    
    Args:
        response_obj: OpenAI response object with choices[0].message
        
    Returns:
        Tuple of (content, reasoning_content)
        - content: Main response content (str, may be empty)
        - reasoning_content: Reasoning content if available (str or None)
    """
    if not response_obj or not hasattr(response_obj, 'choices') or not response_obj.choices:
        return "", None
    
    message = response_obj.choices[0].message
    
    # Extract main content
    content = ""
    if hasattr(message, 'content') and message.content:
        content = message.content.strip()
    
    # Extract reasoning content if available (for reasoning models)
    reasoning_content = None
    if hasattr(message, 'reasoning_content') and message.reasoning_content:
        reasoning_content = message.reasoning_content.strip()
    
    return content, reasoning_content


def remove_think_tags(text: str) -> str:
    """
    Remove <think></think> tags and their content from text.
    
    This is useful for extracting the actual response when reasoning models
    include their reasoning process in <think> tags within the content.
    
    Note: Uses non-greedy regex matching which is efficient for typical LLM
    responses (<10KB). For very large inputs (>1MB), consider preprocessing
    to split into smaller chunks.
    
    Args:
        text: Text that may contain <think></think> tags
        
    Returns:
        Text with <think></think> sections removed
    """
    if not text:
        return ""
    
    # Remove <think>...</think> blocks (non-greedy, multiline, case-insensitive)
    # Use re.DOTALL to match newlines inside the tags
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace that may result from removing think tags
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Replace multiple blank lines with double newline
    
    return cleaned.strip()


def extract_outside_think_tags(text: str) -> str:
    """
    Extract content that appears OUTSIDE of <think></think> tags.
    
    This is the primary function for extracting actionable content from
    reasoning model responses that may include thinking process.
    
    Args:
        text: Text that may contain <think></think> tags
        
    Returns:
        Content outside <think></think> tags, with tags and their content removed
    """
    return remove_think_tags(text)


def get_safe_response_content(content: Optional[str], 
                               reasoning_content: Optional[str],
                               fallback_value: str = "",
                               warn: bool = True) -> str:
    """
    Get safe response content with fallback for None/empty responses.
    
    Priority:
    1. Use content if available
    2. Extract from reasoning_content if content is None/empty
    3. Use fallback_value if both are None/empty
    
    Args:
        content: Main response content
        reasoning_content: Reasoning content (may contain <think> tags)
        fallback_value: Default value to return if both are None/empty
        warn: Whether to issue a warning when using fallback
        
    Returns:
        Safe response content string
    """
    # Try main content first
    if content and content.strip():
        # Remove think tags if present in content
        return extract_outside_think_tags(content)
    
    # Try reasoning content (extract outside think tags)
    if reasoning_content and reasoning_content.strip():
        extracted = extract_outside_think_tags(reasoning_content)
        if extracted:
            if warn:
                warnings.warn(
                    "Main content was empty, extracted from reasoning_content. "
                    "This may indicate a reasoning model response.",
                    UserWarning
                )
            return extracted
    
    # Both are None/empty, use fallback
    if warn:
        warnings.warn(
            f"LLM returned None or empty response. Using fallback value: '{fallback_value[:50]}...'",
            UserWarning
        )
    
    return fallback_value


def parse_llm_response_safely(response_obj,
                               fallback_value: str = "",
                               warn: bool = True) -> str:
    """
    Safely parse LLM response object with comprehensive fallback handling.
    
    This is the main entry point for parsing LLM responses with:
    - Reasoning model support (reasoning_content field)
    - <think></think> tag handling
    - Safe fallback for None/empty responses
    
    Args:
        response_obj: OpenAI response object
        fallback_value: Default value if response is None/empty
        warn: Whether to issue warnings
        
    Returns:
        Parsed response content string
    """
    content, reasoning_content = extract_content_from_response(response_obj)
    return get_safe_response_content(content, reasoning_content, fallback_value, warn)
