"""
Integration test for response parser with gradient agent and optimizer.

Tests that the response parser correctly handles reasoning model responses
in the context of actual gradient agent and optimizer usage.
"""

import sys
import json
import warnings


def test_gradient_agent_with_think_tags():
    """Test gradient agent handling of responses with think tags."""
    from gradient_agent import extract_json_from_text
    from response_parser import extract_outside_think_tags
    
    print("=" * 80)
    print("Testing Gradient Agent with <think> tags")
    print("=" * 80)
    
    # Simulate LLM response with think tags
    llm_response = """<think>
Let me analyze the statistics:
- Overestimated samples: 5
- Underestimated samples: 3
- Mean error: +0.45

I should suggest making the criteria stricter.
</think>

{
    "modifications": [
        {
            "action": "edit",
            "section_name": "Scoring Criteria",
            "opti_direction": "评分标准应该更严格，降低高估倾向"
        }
    ]
}
"""
    
    print("\n### Input (with think tags):")
    print(llm_response)
    
    # Extract content outside think tags
    cleaned_response = extract_outside_think_tags(llm_response)
    print("\n### After removing think tags:")
    print(cleaned_response)
    
    # Extract JSON
    json_text = extract_json_from_text(cleaned_response)
    print("\n### Extracted JSON:")
    print(json_text)
    
    # Parse JSON
    try:
        result = json.loads(json_text)
        print("\n### Parsed successfully:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Verify structure
        assert "modifications" in result
        assert len(result["modifications"]) > 0
        assert result["modifications"][0]["action"] == "edit"
        print("\n✓ Gradient agent test passed")
        return True
    except json.JSONDecodeError as e:
        print(f"\n✗ Failed to parse JSON: {e}")
        return False


def test_optimizer_with_think_tags():
    """Test optimizer handling of responses with think tags."""
    from response_parser import extract_outside_think_tags
    
    print("\n" + "=" * 80)
    print("Testing Optimizer with <think> tags")
    print("=" * 80)
    
    # Simulate optimizer LLM response with think tags
    llm_response = """<think>
The current criteria are:
- Completeness
- Clarity  
- Accuracy

The gradient suggests making it stricter. I should add more specific requirements
and raise the bar for each criterion.
</think>

根据完整性、清晰度、准确性和专业性评估响应。

对于完整性：
- 必须涵盖所有关键要点
- 不能遗漏重要信息

对于清晰度：
- 表达必须清晰明了
- 结构必须有条理

对于准确性：
- 事实必须准确无误
- 不能有明显错误

对于专业性：
- 使用专业术语恰当
- 展现深度理解
"""
    
    print("\n### Input (with think tags):")
    print(llm_response[:200] + "...")
    
    # Extract content outside think tags
    cleaned_response = extract_outside_think_tags(llm_response)
    print("\n### After removing think tags:")
    print(cleaned_response)
    
    # Verify think content is removed
    assert "<think>" not in cleaned_response.lower()
    assert "current criteria" not in cleaned_response
    assert "根据完整性" in cleaned_response
    
    print("\n✓ Optimizer test passed")
    return True


def test_empty_response_handling():
    """Test handling of empty responses with fallback."""
    from gradient_agent import GradientAgent
    import numpy as np
    
    print("\n" + "=" * 80)
    print("Testing Empty Response Handling")
    print("=" * 80)
    
    # Mock LLM function that returns empty response
    def mock_empty_llm(prompt):
        return ""
    
    # Create gradient agent with mock
    agent = GradientAgent(llm_fn=mock_empty_llm, n_samples_per_category=2)
    
    # Generate sample data
    judge_scores = np.array([8.0, 7.5, 6.0, 5.5, 4.0])
    human_scores = np.array([6.0, 6.5, 6.0, 5.5, 5.0])
    responses = ["resp1", "resp2", "resp3", "resp4", "resp5"]
    
    editable_sections = ["Scoring Criteria", "Examples"]
    meta_sections = ["Output Format"]
    
    print("\n### Testing with empty LLM response...")
    
    # This should trigger fallback gradient
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        result = agent.compute_gradient(
            current_prompt="Test prompt",
            editable_sections=editable_sections,
            meta_sections=meta_sections,
            judge_scores=judge_scores,
            human_scores=human_scores,
            responses=responses,
            current_lr=0.1
        )
        
        # Should have warning
        warning_messages = [str(warning.message) for warning in w]
        print(f"\n### Warnings issued: {len(w)}")
        for msg in warning_messages:
            print(f"  - {msg}")
        
        # Should have fallback gradient
        assert "gradient" in result
        assert "modifications" in result["gradient"]
        print(f"\n### Fallback gradient: {result['gradient']}")
        print("\n✓ Empty response handling test passed")
    
    return True


def test_think_only_response():
    """Test response that only contains think tags (no actual content)."""
    from response_parser import extract_outside_think_tags, get_safe_response_content
    
    print("\n" + "=" * 80)
    print("Testing Think-Only Response")
    print("=" * 80)
    
    # Response with only think tags
    think_only = "<think>I'm thinking about this problem...</think>"
    
    result = extract_outside_think_tags(think_only)
    print(f"\n### Input: {think_only}")
    print(f"### Output: '{result}'")
    print(f"### Output is empty: {not result or not result.strip()}")
    
    # Should be empty after extraction
    assert not result or not result.strip()
    
    # Test with safe content getter
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        safe_result = get_safe_response_content(
            content=result,
            reasoning_content=None,
            fallback_value="FALLBACK",
            warn=True
        )
        
        assert safe_result == "FALLBACK"
        assert len(w) == 1
        print(f"### Safe result: {safe_result}")
        print(f"### Warning: {w[0].message}")
    
    print("\n✓ Think-only response test passed")
    return True


def run_all_tests():
    """Run all integration tests."""
    tests = [
        ("Gradient Agent with Think Tags", test_gradient_agent_with_think_tags),
        ("Optimizer with Think Tags", test_optimizer_with_think_tags),
        ("Empty Response Handling", test_empty_response_handling),
        ("Think-Only Response", test_think_only_response),
    ]
    
    print("\n" + "=" * 80)
    print("Running Integration Tests")
    print("=" * 80 + "\n")
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "=" * 80)
        print("All integration tests passed! ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Some tests failed! ✗")
        print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
