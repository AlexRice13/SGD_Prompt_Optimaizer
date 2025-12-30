"""
Gradient Agent: Construct simple optimization direction.

Outputs simple gradient signals with optimization direction and target section,
rather than complex structured schemas.
"""

from typing import List, Dict, Tuple, Callable
import numpy as np
import json
import re
from prompts import GRADIENT_AGENT_SIMPLE_PROMPT_TEMPLATE, format_samples_category
from constants import STRUCTURAL_EDIT_LR_THRESHOLD, can_perform_structural_edit


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON content from text that may contain additional formatting.
    
    Uses multiple strategies to find JSON in the text, handling cases where
    LLMs return JSON wrapped in markdown code blocks or with surrounding text.
    
    Args:
        text: Raw text that may contain JSON
        
    Returns:
        Extracted JSON string, or original text if no JSON pattern found
    """
    # Remove markdown code blocks if present
    text = text.strip()
    if text.startswith('```'):
        # Remove ```json or ``` at start and ``` at end
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()
    
    # Strategy 1: Try to find the first complete JSON object or array
    # Look for opening brace/bracket and match to closing
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        
        # Count braces to find matching closing brace
        count = 0
        for i in range(start_idx, len(text)):
            if text[i] == start_char:
                count += 1
            elif text[i] == end_char:
                count -= 1
                if count == 0:
                    # Found matching closing brace
                    return text[start_idx:i+1]
    
    # Strategy 2: If no balanced JSON found, try regex on cleaned text
    # This handles simple cases
    json_pattern = r'(\{[^}]*\}|\[[^\]]*\])'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Strategy 3: Return original text if nothing found
    return text


class GradientAgent:
    """
    Constructs simple optimization gradient.
    
    Outputs gradient as a simple JSON with just two fields:
    - opti_direction: Abstract optimization direction as a string
    - section_to_opti: Which section should be optimized
    
    This is a simplified version that avoids complex structured schemas.
    """
    
    def __init__(self, llm_fn: Callable[[str], str], 
                 n_samples_per_category: int = 3):
        """
        Initialize gradient agent.
        
        Args:
            llm_fn: Function that takes a prompt and returns JSON-formatted text
            n_samples_per_category: Number of samples to select from each category
        """
        self.llm_fn = llm_fn
        self.n_samples_per_category = n_samples_per_category
        # Track recently modified sections for diversity (cooldown mechanism)
        self.modification_history = []  # List of (step, section_name) tuples
        self.current_step = 0
    
    def select_gradient_samples(self, judge_scores: np.ndarray, 
                               human_scores: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Select samples for gradient construction.
        
        Selects three categories:
        1. Overestimated: JudgeScore >> HumanScore
        2. Underestimated: JudgeScore << HumanScore
        3. Well-aligned: JudgeScore ≈ HumanScore
        
        Args:
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            
        Returns:
            Dictionary with indices for each category
        """
        errors = judge_scores - human_scores
        abs_errors = np.abs(errors)
        
        # Category 1: Overestimated (positive error)
        overestimated_idx = np.where(errors > 0)[0]
        if len(overestimated_idx) > 0:
            # Sort by error magnitude
            sorted_idx = overestimated_idx[np.argsort(-errors[overestimated_idx])]
            overestimated = sorted_idx[:self.n_samples_per_category]
        else:
            overestimated = np.array([])
        
        # Category 2: Underestimated (negative error)
        underestimated_idx = np.where(errors < 0)[0]
        if len(underestimated_idx) > 0:
            # Sort by error magnitude
            sorted_idx = underestimated_idx[np.argsort(errors[underestimated_idx])]
            underestimated = sorted_idx[:self.n_samples_per_category]
        else:
            underestimated = np.array([])
        
        # Category 3: Well-aligned (small error)
        sorted_by_abs_error = np.argsort(abs_errors)
        well_aligned = sorted_by_abs_error[:self.n_samples_per_category]
        
        return {
            'overestimated': overestimated,
            'underestimated': underestimated,
            'well_aligned': well_aligned
        }
    
    def compute_statistics(self, judge_scores: np.ndarray, 
                          human_scores: np.ndarray,
                          selected_indices: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Compute aggregated statistics (information isolation constraint).
        
        Returns only aggregated statistics, not individual sample details.
        
        Args:
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            selected_indices: Indices for each category
            
        Returns:
            Aggregated statistics dictionary
        """
        errors = judge_scores - human_scores
        
        stats = {
            'total_samples': len(judge_scores),
            'overestimated': {
                'count': len(selected_indices['overestimated']),
                'mean_error': float(np.mean(errors[selected_indices['overestimated']])) 
                             if len(selected_indices['overestimated']) > 0 else 0.0,
                'max_error': float(np.max(errors[selected_indices['overestimated']]))
                            if len(selected_indices['overestimated']) > 0 else 0.0,
            },
            'underestimated': {
                'count': len(selected_indices['underestimated']),
                'mean_error': float(np.mean(errors[selected_indices['underestimated']]))
                             if len(selected_indices['underestimated']) > 0 else 0.0,
                'min_error': float(np.min(errors[selected_indices['underestimated']]))
                            if len(selected_indices['underestimated']) > 0 else 0.0,
            },
            'well_aligned': {
                'count': len(selected_indices['well_aligned']),
                'mean_abs_error': float(np.mean(np.abs(errors[selected_indices['well_aligned']])))
                                 if len(selected_indices['well_aligned']) > 0 else 0.0,
            },
            'overall': {
                'mae': float(np.mean(np.abs(errors))),
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
            }
        }
        
        return stats
    
    def update_modification_history(self, modified_sections: List[str]):
        """
        Update the history of recently modified sections for diversity tracking.
        
        Args:
            modified_sections: List of section names that were modified in this step
        """
        self.current_step += 1
        for section in modified_sections:
            self.modification_history.append((self.current_step, section))
        
        # Keep only last 5 steps in history to avoid memory bloat
        cutoff_step = self.current_step - 5
        self.modification_history = [
            (step, section) for step, section in self.modification_history
            if step > cutoff_step
        ]
    
    def get_diversity_hint(self, editable_sections: List[str]) -> str:
        """
        Generate a diversity hint based on recent modification history.
        
        Identifies sections that were modified recently and suggests avoiding them
        to encourage exploration of different optimization dimensions.
        
        Args:
            editable_sections: List of currently editable sections
            
        Returns:
            String hint for diversity (empty if no recent history)
        """
        if not self.modification_history or len(editable_sections) <= 1:
            return ""
        
        # Count modifications in recent history (last 3 steps)
        recent_cutoff = self.current_step - 3
        recent_mods = [
            section for step, section in self.modification_history
            if step > recent_cutoff
        ]
        
        if not recent_mods:
            return ""
        
        # Find sections modified multiple times recently
        from collections import Counter
        mod_counts = Counter(recent_mods)
        frequently_modified = [
            section for section, count in mod_counts.items()
            if count >= 2 and section in editable_sections
        ]
        
        if frequently_modified:
            sections_str = '、'.join(frequently_modified)
            return (
                f"\n**多样性提示（Diversity Hint）**：\n"
                f"以下sections在最近几步中已被频繁修改：{sections_str}\n"
                f"建议考虑优化其他sections以实现多维度改进，避免局部最优。"
            )
        
        return ""
    
    def construct_structured_gradient(self, current_prompt: str,
                                      editable_sections: List[str],
                                      meta_sections: List[str],
                                      statistics: Dict[str, any],
                                      selected_indices: Dict[str, np.ndarray],
                                      judge_scores: np.ndarray,
                                      human_scores: np.ndarray,
                                      responses: List[str],
                                      current_lr: float) -> Dict:
        """
        Construct gradient with multiple section modifications.
        
        Outputs a list of modifications, each containing:
        - action: "edit", "add", or "remove"
        - section_name: Which section to operate on
        - opti_direction: Optimization direction (for edit/add actions)
        
        Args:
            current_prompt: Current JudgePrompt text
            editable_sections: List of sections that can be modified
            meta_sections: List of immutable sections
            statistics: Aggregated statistics
            selected_indices: Selected sample indices by category
            judge_scores: All judge scores
            human_scores: All human scores
            responses: All response texts
            current_lr: Current learning rate
            
        Returns:
            Gradient dictionary with list of modifications
        """
        # Build sample content strings for LLM context using centralized formatter
        overestimated_samples = format_samples_category(
            selected_indices['overestimated'], responses, human_scores, 
            judge_scores, "高估", self.n_samples_per_category
        )
        underestimated_samples = format_samples_category(
            selected_indices['underestimated'], responses, human_scores,
            judge_scores, "低估", self.n_samples_per_category
        )
        well_aligned_samples = format_samples_category(
            selected_indices['well_aligned'], responses, human_scores,
            judge_scores, "对齐", self.n_samples_per_category
        )
        
        # Get diversity hint based on recent modification history
        diversity_hint = self.get_diversity_hint(editable_sections)
        
        # Use centralized prompt template (will be updated for simplified output)
        from prompts import GRADIENT_AGENT_SIMPLE_PROMPT_TEMPLATE
        gradient_prompt = GRADIENT_AGENT_SIMPLE_PROMPT_TEMPLATE.format(
            current_prompt=current_prompt,
            editable_sections=', '.join(editable_sections),
            meta_sections=', '.join(meta_sections),
            total_samples=statistics['total_samples'],
            overall_mae=f"{statistics['overall']['mae']:.3f}",
            mean_error=f"{statistics['overall']['mean_error']:.3f}",
            std_error=f"{statistics['overall']['std_error']:.3f}",
            overestimated_count=statistics['overestimated']['count'],
            overestimated_mean_error=f"+{statistics['overestimated']['mean_error']:.3f}",
            underestimated_count=statistics['underestimated']['count'],
            underestimated_mean_error=f"{statistics['underestimated']['mean_error']:.3f}",
            well_aligned_count=statistics['well_aligned']['count'],
            well_aligned_mae=f"{statistics['well_aligned']['mean_abs_error']:.3f}",
            overestimated_samples=overestimated_samples,
            underestimated_samples=underestimated_samples,
            well_aligned_samples=well_aligned_samples,
            diversity_hint=diversity_hint,
            current_lr=f"{current_lr:.4f}",
            structural_lr_threshold=f"{STRUCTURAL_EDIT_LR_THRESHOLD:.1f}"
        )

        # Call LLM to get gradient with multiple modifications
        llm_output = self.llm_fn(gradient_prompt)
        
        # Parse JSON (with error handling)
        try:
            # Extract JSON from text using regex-based extraction
            json_text = extract_json_from_text(llm_output)
            
            # Additional cleaning: fix common JSON issues
            # Replace Python-style True/False with JSON true/false
            json_text = json_text.replace(': True', ': true')
            json_text = json_text.replace(': False', ': false')
            json_text = json_text.replace(':True', ':true')
            json_text = json_text.replace(':False', ':false')
            
            # Remove trailing commas before closing braces/brackets
            json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
            
            gradient_result = json.loads(json_text)
            
            # Validate schema - should have modifications array
            if 'modifications' not in gradient_result:
                raise ValueError("Missing required field: modifications")
            
            modifications = gradient_result['modifications']
            if not isinstance(modifications, list):
                raise ValueError("modifications must be a list")
            
            # Validate and filter modifications based on LR threshold
            validated_modifications = []
            
            for mod in modifications:
                # Validate required fields
                if 'action' not in mod or 'section_name' not in mod:
                    print(f"Warning: Skipping modification with missing action or section_name: {mod}")
                    continue
                
                action = mod['action']
                section_name = mod['section_name']
                
                # Validate action type
                if action not in ['edit', 'add', 'remove']:
                    print(f"Warning: Invalid action '{action}', skipping")
                    continue
                
                # Check LR threshold for structural changes
                if action in ['add', 'remove'] and not can_perform_structural_edit(current_lr):
                    print(f"Warning: Cannot {action} section at LR={current_lr:.4f} < {STRUCTURAL_EDIT_LR_THRESHOLD}, skipping")
                    continue
                
                # Check meta section constraint
                if action == 'edit' and section_name in meta_sections:
                    print(f"Warning: Cannot edit meta section '{section_name}', skipping")
                    continue
                
                if action == 'remove' and section_name in meta_sections:
                    print(f"Warning: Cannot remove meta section '{section_name}', skipping")
                    continue
                
                if action == 'add' and section_name in meta_sections:
                    print(f"Warning: Cannot add section with meta section name '{section_name}', skipping")
                    continue
                
                # Validate opti_direction for edit/add actions
                if action in ['edit', 'add'] and 'opti_direction' not in mod:
                    print(f"Warning: Missing opti_direction for {action} action, skipping")
                    continue
                
                validated_modifications.append(mod)
            
            # If no valid modifications, return fallback
            if not validated_modifications:
                print("Warning: No valid modifications after validation, using fallback")
                return self._get_fallback_gradient(editable_sections, statistics, current_lr)
            
            return {'modifications': validated_modifications}
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to default structure if parsing fails
            print(f"Warning: Failed to parse gradient: {e}")
            print(f"LLM output preview: {llm_output[:200]}...")
            return self._get_fallback_gradient(editable_sections, statistics, current_lr)
    
    def _get_fallback_gradient(self, editable_sections: List[str], 
                               statistics: Dict, current_lr: float) -> Dict:
        """Generate a safe fallback gradient if LLM output fails."""
        # Determine primary error mode and direction
        mean_error = statistics['overall']['mean_error']
        if mean_error > 0.1:
            direction = "评分标准应该更严格，降低高估倾向"
        elif mean_error < -0.1:
            direction = "评分标准应该更宽松，降低低估倾向"
        else:
            direction = "保持当前评分标准，轻微调整以减少评分波动"
        
        # Select first editable section as target
        target_section = editable_sections[0] if editable_sections else "Scoring Criteria"
        
        return {
            "modifications": [
                {
                    "action": "edit",
                    "section_name": target_section,
                    "opti_direction": direction
                }
            ]
        }
    
    def compute_gradient(self, current_prompt: str,
                        editable_sections: List[str],
                        meta_sections: List[str],
                        judge_scores: np.ndarray,
                        human_scores: np.ndarray,
                        responses: List[str],
                        current_lr: float) -> Dict:
        """
        Full gradient computation pipeline - multi-section version.
        
        Args:
            current_prompt: Current JudgePrompt text
            editable_sections: List of editable section names
            meta_sections: List of immutable section names
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            responses: List of response texts for reference
            current_lr: Current learning rate
            
        Returns:
            Dictionary containing gradient with list of modifications
        """
        # Select samples
        selected_indices = self.select_gradient_samples(judge_scores, human_scores)
        
        # Compute aggregated statistics
        statistics = self.compute_statistics(judge_scores, human_scores, selected_indices)
        
        # Construct gradient with multiple modifications
        gradient = self.construct_structured_gradient(
            current_prompt, editable_sections, meta_sections,
            statistics, selected_indices,
            judge_scores, human_scores, responses,
            current_lr
        )
        
        return {
            'statistics': statistics,
            'selected_indices': selected_indices,
            'gradient': gradient
        }


if __name__ == '__main__':
    """Unit tests for GradientAgent class."""
    
    print("Running GradientAgent unit tests...")
    
    # Test 1: Basic initialization
    print("\n1. Testing basic initialization...")
    
    def mock_gradient_llm(prompt):
        """Mock gradient LLM function."""
        return '{"modifications": [{"action": "edit", "section_name": "Test", "opti_direction": "Improve"}]}'
    
    agent = GradientAgent(mock_gradient_llm, n_samples_per_category=3)
    assert agent.n_samples_per_category == 3
    assert agent.llm_fn == mock_gradient_llm
    print("   ✓ Initialization works")
    
    # Test 2: Parse gradient response
    print("\n2. Testing parse_gradient_response...")
    valid_json = '{"modifications": [{"action": "edit", "section_name": "S1", "opti_direction": "Better"}]}'
    parsed = agent.parse_gradient_response(valid_json)
    assert parsed is not None
    assert 'modifications' in parsed
    assert len(parsed['modifications']) == 1
    print("   ✓ Valid JSON parsing works")
    
    # Test 3: Invalid JSON
    print("\n3. Testing invalid JSON handling...")
    invalid_json = "Not valid JSON"
    parsed_invalid = agent.parse_gradient_response(invalid_json)
    assert parsed_invalid is None
    print("   ✓ Invalid JSON handled gracefully")
    
    # Test 4: Empty modifications
    print("\n4. Testing empty modifications...")
    empty_json = '{"modifications": []}'
    parsed_empty = agent.parse_gradient_response(empty_json)
    assert parsed_empty is not None
    assert 'modifications' in parsed_empty
    assert len(parsed_empty['modifications']) == 0
    print("   ✓ Empty modifications handled")
    
    print("\n" + "="*50)
    print("All GradientAgent tests passed! ✓")
    print("="*50)
