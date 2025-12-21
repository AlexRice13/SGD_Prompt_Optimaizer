"""
Gradient Agent: Construct structured semantic pressure tensor.

Outputs structured gradient signals that are isomorphic to the optimizer's
action space, rather than natural language suggestions.
"""

from typing import List, Dict, Tuple, Callable
import numpy as np
import json
import re
from prompts import GRADIENT_AGENT_PROMPT_TEMPLATE, format_samples_category


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
    Constructs structured semantic pressure tensor.
    
    Outputs gradient as a structured schema aligned with optimizer's action space,
    rather than natural language suggestions. Each pressure signal explicitly
    specifies section_id, action_type, direction, and magnitude.
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
        
        # Define valid enumeration values for schema
        self.valid_pressure_types = {
            'constraint_strictness', 'evaluation_threshold', 
            'preference_weight', 'ambiguity_tolerance'
        }
        self.valid_directions = {'increase', 'decrease', 'maintain'}
        self.valid_error_modes = {'overestimation', 'underestimation', 'variance'}
        self.valid_magnitudes = {'weak', 'medium', 'strong'}
        self.valid_confidences = {'low', 'medium', 'high'}
    
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
    
    def construct_structured_gradient(self, current_prompt: str,
                                      editable_sections: List[str],
                                      meta_sections: List[str],
                                      statistics: Dict[str, any],
                                      selected_indices: Dict[str, np.ndarray],
                                      judge_scores: np.ndarray,
                                      human_scores: np.ndarray,
                                      responses: List[str],
                                      current_lr: float,
                                      structural_edit_threshold: float) -> Dict:
        """
        Construct structured semantic pressure tensor.
        
        Outputs a schema-aligned gradient instead of natural language.
        
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
            structural_edit_threshold: LR threshold for structural edits
            
        Returns:
            Structured gradient dictionary
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
        
        # Determine action permissions based on LR
        can_add_section = current_lr >= structural_edit_threshold
        can_delete_section = current_lr >= structural_edit_threshold
        can_modify_content = True
        
        # Use centralized prompt template
        gradient_prompt = GRADIENT_AGENT_PROMPT_TEMPLATE.format(
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
            can_add_section=can_add_section,
            can_delete_section=can_delete_section,
            can_modify_content=can_modify_content,
            current_lr=f"{current_lr:.4f}",
            structural_edit_threshold=f"{structural_edit_threshold:.4f}"
        )

        # Call LLM to get structured output
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
            
            structured_gradient = json.loads(json_text)
            
            # Validate schema
            self._validate_gradient_schema(structured_gradient, editable_sections)
            
            return structured_gradient
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to default structure if parsing fails
            print(f"Warning: Failed to parse structured gradient: {e}")
            print(f"LLM output preview: {llm_output[:200]}...")
            return self._get_fallback_gradient(editable_sections, statistics)
    
    def _validate_gradient_schema(self, gradient: Dict, editable_sections: List[str]):
        """Validate that gradient follows required schema."""
        required_keys = {'global_signals', 'section_pressures', 'acknowledged_action_space'}
        if not required_keys.issubset(gradient.keys()):
            raise ValueError(f"Missing required keys: {required_keys - set(gradient.keys())}")
        
        # Validate section_pressures
        for pressure in gradient['section_pressures']:
            if pressure['section_id'] not in editable_sections:
                print(f"Warning: Unknown section_id '{pressure['section_id']}' in gradient")
            
            required_fields = {'section_id', 'pressure_type', 'direction', 
                             'affected_error_mode', 'magnitude_bucket', 'confidence'}
            if not required_fields.issubset(pressure.keys()):
                raise ValueError(f"Section pressure missing fields: {required_fields - set(pressure.keys())}")
            
            # Validate enumerations
            if pressure['pressure_type'] not in self.valid_pressure_types:
                raise ValueError(f"Invalid pressure_type: {pressure['pressure_type']}")
            if pressure['direction'] not in self.valid_directions:
                raise ValueError(f"Invalid direction: {pressure['direction']}")
            if pressure['affected_error_mode'] not in self.valid_error_modes:
                raise ValueError(f"Invalid error_mode: {pressure['affected_error_mode']}")
            if pressure['magnitude_bucket'] not in self.valid_magnitudes:
                raise ValueError(f"Invalid magnitude: {pressure['magnitude_bucket']}")
            if pressure['confidence'] not in self.valid_confidences:
                raise ValueError(f"Invalid confidence: {pressure['confidence']}")
    
    def _get_fallback_gradient(self, editable_sections: List[str], 
                              statistics: Dict) -> Dict:
        """Generate a safe fallback gradient if LLM output fails."""
        # Determine primary error mode
        mean_error = statistics['overall']['mean_error']
        if mean_error > 0.1:
            primary_mode = "overestimation"
            direction = "increase"  # Increase constraint strictness
        elif mean_error < -0.1:
            primary_mode = "underestimation"
            direction = "decrease"  # Decrease constraint strictness
        else:
            primary_mode = "variance"
            direction = "maintain"
        
        # Create minimal valid gradient
        section_pressures = []
        for section in editable_sections:
            section_pressures.append({
                "section_id": section,
                "immutability_ack": False,
                "pressure_type": "constraint_strictness",
                "direction": direction,
                "affected_error_mode": primary_mode,
                "magnitude_bucket": "weak",
                "confidence": "low"
            })
        
        return {
            "global_signals": {
                "bias_direction": "up" if mean_error > 0 else "down" if mean_error < 0 else "neutral",
                "variance_pressure": "tighten" if statistics['overall']['std_error'] > 1.0 else "stable"
            },
            "section_pressures": section_pressures,
            "acknowledged_action_space": {
                "allow_add_section": False,
                "allow_delete_section": False,
                "allow_sentence_edit": True,
                "allow_token_edit": True
            },
            "conflicting_pressures": [],
            "redundancy_groups": []
        }
    
    def compute_gradient(self, current_prompt: str,
                        editable_sections: List[str],
                        meta_sections: List[str],
                        judge_scores: np.ndarray,
                        human_scores: np.ndarray,
                        responses: List[str],
                        current_lr: float,
                        structural_edit_threshold: float) -> Dict:
        """
        Full gradient computation pipeline.
        
        Args:
            current_prompt: Current JudgePrompt text
            editable_sections: List of editable section names
            meta_sections: List of immutable section names
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            responses: List of response texts for reference
            current_lr: Current learning rate
            structural_edit_threshold: LR threshold for structural edits
            
        Returns:
            Dictionary containing structured gradient
        """
        # Select samples
        selected_indices = self.select_gradient_samples(judge_scores, human_scores)
        
        # Compute aggregated statistics
        statistics = self.compute_statistics(judge_scores, human_scores, selected_indices)
        
        # Construct structured gradient
        structured_gradient = self.construct_structured_gradient(
            current_prompt, editable_sections, meta_sections,
            statistics, selected_indices,
            judge_scores, human_scores, responses,
            current_lr, structural_edit_threshold
        )
        
        return {
            'statistics': statistics,
            'selected_indices': selected_indices,
            'structured_gradient': structured_gradient
        }
