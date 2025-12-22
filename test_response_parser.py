"""
Test script for response parser functionality.

This script tests the response_parser module to ensure:
1. Think tag removal works correctly
2. Content extraction handles various formats
3. Safe fallback works when responses are None/empty
"""

import sys
import warnings

# Mock response object to simulate OpenAI responses
class MockMessage:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content

class MockChoice:
    def __init__(self, message):
        self.message = message

class MockResponse:
    def __init__(self, content=None, reasoning_content=None):
        self.choices = [MockChoice(MockMessage(content, reasoning_content))]


def test_response_parser():
    """Test response parser utilities."""
    from response_parser import (
        extract_content_from_response,
        remove_think_tags,
        extract_outside_think_tags,
        get_safe_response_content,
        parse_llm_response_safely
    )
    
    print("=" * 80)
    print("Testing Response Parser Utilities")
    print("=" * 80)
    
    # Test 1: Remove think tags
    print("\n### Test 1: Remove <think></think> tags")
    text_with_think = """<think>
This is my reasoning process.
I'm thinking about the problem.
</think>

Here is my actual response:
{"modifications": [{"action": "edit", "section_name": "Test"}]}
"""
    result = remove_think_tags(text_with_think)
    print(f"Input:\n{text_with_think}")
    print(f"\nOutput:\n{result}")
    assert "<think>" not in result
    assert "actual response" in result
    print("✓ Test 1 passed")
    
    # Test 2: Extract outside think tags
    print("\n### Test 2: Extract content outside think tags")
    result = extract_outside_think_tags(text_with_think)
    print(f"Extracted: {result}")
    assert "reasoning process" not in result
    assert "actual response" in result
    print("✓ Test 2 passed")
    
    # Test 3: Handle response with reasoning_content
    print("\n### Test 3: Handle reasoning model response")
    mock_response = MockResponse(
        content='{"score": 8.5}',
        reasoning_content='<think>Let me analyze this...</think>'
    )
    content, reasoning = extract_content_from_response(mock_response)
    print(f"Content: {content}")
    print(f"Reasoning: {reasoning}")
    assert content == '{"score": 8.5}'
    assert "think" in reasoning.lower()
    print("✓ Test 3 passed")
    
    # Test 4: Handle None content with fallback
    print("\n### Test 4: Handle None content with fallback")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = get_safe_response_content(None, None, fallback_value="FALLBACK", warn=True)
        assert result == "FALLBACK"
        assert len(w) == 1
        assert "None or empty response" in str(w[0].message)
        print(f"Result: {result}")
        print(f"Warning issued: {w[0].message}")
    print("✓ Test 4 passed")
    
    # Test 5: Extract from reasoning_content when content is empty
    print("\n### Test 5: Extract from reasoning_content when content is empty")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = get_safe_response_content(
            "", 
            "<think>thinking...</think>Actual answer here",
            warn=True
        )
        assert result == "Actual answer here"
        assert len(w) == 1
        assert "extracted from reasoning_content" in str(w[0].message)
        print(f"Result: {result}")
        print(f"Warning issued: {w[0].message}")
    print("✓ Test 5 passed")
    
    # Test 6: Parse full response object safely
    print("\n### Test 6: Parse full response object safely")
    mock_response = MockResponse(
        content=None,
        reasoning_content='<think>Let me think...</think>{"result": "success"}'
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_llm_response_safely(mock_response, warn=True)
        assert result == '{"result": "success"}'
        print(f"Result: {result}")
        if w:
            print(f"Warning issued: {w[0].message}")
    print("✓ Test 6 passed")
    
    # Test 7: Case-insensitive think tag removal
    print("\n### Test 7: Case-insensitive think tag removal")
    text_mixed_case = "<THINK>uppercase thinking</THINK>actual content<Think>more thinking</Think>end"
    result = remove_think_tags(text_mixed_case)
    print(f"Input: {text_mixed_case}")
    print(f"Output: {result}")
    assert "<think>" not in result.lower()
    assert "<THINK>" not in result
    assert "<Think>" not in result
    assert "uppercase thinking" not in result
    assert "more thinking" not in result
    assert "actual content" in result
    assert "end" in result
    print("✓ Test 7 passed")
    
    # Test 8: Multiple think blocks
    print("\n### Test 8: Multiple think blocks")
    text_multiple = """
First response part
<think>first thinking</think>
Middle part
<think>second thinking</think>
Last part
"""
    result = remove_think_tags(text_multiple)
    print(f"Output: {result}")
    assert "thinking" not in result
    assert "First response part" in result
    assert "Middle part" in result
    assert "Last part" in result
    print("✓ Test 8 passed")
    
    # Test 9: Nested content (should handle properly)
    print("\n### Test 9: Nested tags handling")
    text_nested = "<think>outer<inner>nested</inner></think>content"
    result = remove_think_tags(text_nested)
    print(f"Input: {text_nested}")
    print(f"Output: {result}")
    assert result.strip() == "content"
    print("✓ Test 9 passed")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        test_response_parser()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
