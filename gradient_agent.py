"""
Gradient Agent: Construct structured semantic pressure tensor.

Outputs structured gradient signals that are isomorphic to the optimizer's
action space, rather than natural language suggestions.
"""

from typing import List, Dict, Tuple, Callable
import numpy as np
import json


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
        # Build sample content strings for LLM context
        def format_samples(indices, category_name):
            if len(indices) == 0:
                return f"  （无{category_name}样本）\n"
            
            sample_text = ""
            for i, idx in enumerate(indices[:self.n_samples_per_category], 1):
                response = responses[idx] if idx < len(responses) else "[缺失]"
                if len(response) > 200:
                    response = response[:200] + "..."
                error = judge_scores[idx] - human_scores[idx]
                sample_text += f"  样本{i}: {response}\n"
                sample_text += f"    人工评分: {human_scores[idx]:.1f}, AI评分: {judge_scores[idx]:.1f}, 误差: {error:+.1f}\n"
            return sample_text
        
        overestimated_samples = format_samples(selected_indices['overestimated'], "高估")
        underestimated_samples = format_samples(selected_indices['underestimated'], "低估")
        well_aligned_samples = format_samples(selected_indices['well_aligned'], "对齐")
        
        # Determine action permissions based on LR
        can_add_section = current_lr >= structural_edit_threshold
        can_delete_section = current_lr >= structural_edit_threshold
        can_modify_content = True
        
        gradient_prompt = f"""你是一个元优化器，为评分prompt生成**结构化语义压力张量**。

当前评分Prompt的sections：
{current_prompt}

可编辑sections: {', '.join(editable_sections)}
元sections（永不可改）: {', '.join(meta_sections)}

性能统计：
- 总样本: {statistics['total_samples']}
- 整体MAE: {statistics['overall']['mae']:.3f}
- 平均偏置: {statistics['overall']['mean_error']:.3f}
- 误差标准差: {statistics['overall']['std_error']:.3f}

误差分类：
1. 高估（AI>{human_scores.mean():.1f}+）: {statistics['overestimated']['count']}个, 均误差+{statistics['overestimated']['mean_error']:.3f}
2. 低估（AI<{human_scores.mean():.1f}-）: {statistics['underestimated']['count']}个, 均误差{statistics['underestimated']['mean_error']:.3f}
3. 对齐: {statistics['well_aligned']['count']}个, MAE={statistics['well_aligned']['mean_abs_error']:.3f}

参考样本（仅供模式分析，勿在输出中引用）：
高估样本：
{overestimated_samples}
低估样本：
{underestimated_samples}
对齐样本：
{well_aligned_samples}

当前动作空间权限：
- 可添加section: {can_add_section}
- 可删除section: {can_delete_section}
- 可修改内容: {can_modify_content}
- 当前LR: {current_lr:.4f}, 结构编辑阈值: {structural_edit_threshold:.4f}

=== 任务：输出JSON格式的结构化梯度 ===

你必须输出一个JSON对象，包含以下字段：

1. global_signals (全局信号，仅作约束，不直接修改):
   - bias_direction: "up" / "down" / "neutral"
   - variance_pressure: "tighten" / "loosen" / "stable"

2. section_pressures (核心：每个可编辑section的压力块):
   [
     {{
       "section_id": "section名称",
       "immutability_ack": false,  // 必须为false（自检）
       "pressure_type": "constraint_strictness" | "evaluation_threshold" | "preference_weight" | "ambiguity_tolerance",
       "direction": "increase" | "decrease" | "maintain",
       "affected_error_mode": "overestimation" | "underestimation" | "variance",
       "magnitude_bucket": "weak" | "medium" | "strong",
       "confidence": "low" | "medium" | "high"
     }},
     ...
   ]

3. acknowledged_action_space (声明理解的动作边界):
   {{
     "allow_add_section": {can_add_section},
     "allow_delete_section": {can_delete_section},
     "allow_sentence_edit": true,
     "allow_token_edit": true
   }}

4. conflicting_pressures (可选，列出冲突的section pairs):
   [[" section_id_A", "section_id_B"], ...]

5. redundancy_groups (可选，列出可能冗余的section groups):
   [["section_id_X", "section_id_Y"], ...]

=== 严格约束 ===
- 不要输出任何"怎么改""改成什么"的描述性文本
- 不要引用具体样本内容
- 不要给出修改建议的自然语言描述
- 只输出上述JSON结构
- section_id必须精确匹配可编辑sections列表
- 所有枚举值必须从上述选项中选择

基于统计和样本模式，识别每个可编辑section的语义压力方向和强度。
输出纯JSON，不要markdown代码块标记。"""

        # Call LLM to get structured output
        llm_output = self.llm_fn(gradient_prompt)
        
        # Parse JSON (with error handling)
        try:
            # Remove markdown code blocks if present
            llm_output = llm_output.strip()
            if llm_output.startswith('```'):
                lines = llm_output.split('\n')
                llm_output = '\n'.join(lines[1:-1]) if len(lines) > 2 else llm_output
            if llm_output.startswith('```json'):
                llm_output = llm_output[7:]
            if llm_output.endswith('```'):
                llm_output = llm_output[:-3]
            
            structured_gradient = json.loads(llm_output.strip())
            
            # Validate schema
            self._validate_gradient_schema(structured_gradient, editable_sections)
            
            return structured_gradient
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to default structure if parsing fails
            print(f"Warning: Failed to parse structured gradient: {e}")
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
