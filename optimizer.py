"""
Optimizer: Prompt modification with learning rate-based permissions.

Implements the Optimizer agent that generates prompt modification suggestions
based on proxy gradients, with permissions bound to current learning rate.
"""

from typing import Callable, Dict, List, Optional
import difflib


class PromptOptimizer:
    """
    Optimizer that generates prompt modification suggestions.
    
    Permissions are bound to learning rate with threshold ratio:
    - High LR (above threshold ratio): Can add or remove editable sections
    - Low LR (below threshold ratio): Only modify content of existing editable sections
    - Modification character limit scales with LR ratio
    
    Note: meta_sections can NEVER be modified or deleted at any LR stage.
    """
    
    def __init__(self, llm_fn: Callable[[str], str], 
                 structural_edit_threshold_ratio: float = 0.5,
                 initial_lr: float = 0.1):
        """
        Initialize optimizer.
        
        Args:
            llm_fn: Function that takes a prompt and returns text
            structural_edit_threshold_ratio: Ratio of max LR above which structural 
                                             edits (add/remove sections) are allowed.
                                             E.g., 0.5 means structural edits only when
                                             current_lr >= 0.5 * initial_lr
            initial_lr: Initial learning rate to compute threshold
        
        Raises:
            ValueError: If initial_lr <= 0
        """
        if initial_lr <= 0:
            raise ValueError(f"initial_lr must be positive, got {initial_lr}")
        
        self.llm_fn = llm_fn
        self.structural_edit_threshold_ratio = structural_edit_threshold_ratio
        self.initial_lr = initial_lr
        self.structural_edit_threshold = initial_lr * structural_edit_threshold_ratio
    
    def get_permissions(self, learning_rate: float) -> Dict[str, bool]:
        """
        Determine modification permissions based on learning rate.
        
        Args:
            learning_rate: Current learning rate
            
        Returns:
            Dictionary of permission flags
        """
        # Structural edits only allowed when LR is above threshold
        is_high_lr = learning_rate >= self.structural_edit_threshold
        
        return {
            'modify_content': True,  # Always allowed for editable sections
            'add_sections': is_high_lr,  # Can add new editable sections
            'remove_sections': is_high_lr,  # Can remove editable sections
            'reorder_sections': False,  # Not currently supported
        }
    
    def compute_max_chars(self, learning_rate: float, base_limit: int = 100) -> int:
        """
        Compute maximum character changes based on learning rate ratio.
        
        Character limit scales with (current_lr / initial_lr) ratio,
        so modifications become more constrained as training progresses.
        Note: LR ratio is clamped to [0, 1] range, so if LR exceeds initial_lr
        (which can happen in some schedulers), the char limit is capped at base_limit.
        
        Args:
            learning_rate: Current learning rate
            base_limit: Base character limit at initial LR
            
        Returns:
            Maximum allowed character changes
        """
        # Scale character limit with LR ratio relative to initial LR
        lr_ratio = learning_rate / self.initial_lr
        lr_ratio = max(0.0, min(1.0, lr_ratio))  # Clamp to [0, 1]
        
        return max(10, int(base_limit * lr_ratio))
    
    def generate_modification_from_structured_gradient(self, 
                                                       current_prompt: str,
                                                       structured_gradient: Dict,
                                                       learning_rate: float,
                                                       editable_sections: List[str],
                                                       meta_sections: List[str]) -> str:
        """
        Generate prompt modification from structured gradient.
        
        Args:
            current_prompt: Current JudgePrompt text
            structured_gradient: Structured gradient dictionary from GradientAgent
            learning_rate: Current learning rate
            editable_sections: List of sections that can be modified
            meta_sections: List of sections that cannot be modified or deleted
            
        Returns:
            Modification suggestion as text
        """
        # Validate action space acknowledgment
        ack = structured_gradient.get('acknowledged_action_space', {})
        permissions = self.get_permissions(learning_rate)
        
        # Check consistency
        if ack.get('allow_add_section') != permissions['add_sections']:
            print(f"Warning: Gradient action space mismatch on add_sections")
        if ack.get('allow_delete_section') != permissions['remove_sections']:
            print(f"Warning: Gradient action space mismatch on remove_sections")
        
        # Extract high-confidence pressures
        section_pressures = structured_gradient.get('section_pressures', [])
        high_conf_pressures = [p for p in section_pressures 
                               if p.get('confidence') in ['medium', 'high']]
        
        if not high_conf_pressures:
            # No confident pressures, skip modification
            return "NO_MODIFICATION"
        
        # Select pressure with highest magnitude (strong > medium > weak)
        magnitude_priority = {'strong': 3, 'medium': 2, 'weak': 1}
        high_conf_pressures.sort(
            key=lambda p: magnitude_priority.get(p.get('magnitude_bucket', 'weak'), 0),
            reverse=True
        )
        
        primary_pressure = high_conf_pressures[0]
        
        # Map structured pressure to modification instruction
        max_chars = self.compute_max_chars(learning_rate)
        
        # Build modification prompt based on pressure type
        pressure_type = primary_pressure.get('pressure_type')
        direction = primary_pressure.get('direction')
        section_id = primary_pressure.get('section_id')
        error_mode = primary_pressure.get('affected_error_mode')
        magnitude = primary_pressure.get('magnitude_bucket')
        
        # Map pressure to modification guidance
        modification_guidance = self._map_pressure_to_guidance(
            pressure_type, direction, error_mode, magnitude
        )
        
        optimizer_prompt = f"""你是一个prompt优化代理。为评分Prompt生成修改建议。

当前评分Prompt：
{current_prompt}

结构化梯度信号：
- 目标section: {section_id}
- 压力类型: {pressure_type}
- 方向: {direction}
- 受影响的误差模式: {error_mode}
- 强度: {magnitude}

修改指导：
{modification_guidance}

优化约束：
- 学习率: {learning_rate:.4f}
- 最大字符修改数: {max_chars}
- 可编辑sections: {', '.join(editable_sections)}
- 元sections（不可修改或删除）: {', '.join(meta_sections)}

权限：
- 修改可编辑sections的内容: {permissions['modify_content']}
- 添加新sections: {permissions['add_sections']}
- 删除可编辑sections: {permissions['remove_sections']}

重要规则：
- 元sections（{', '.join(meta_sections)}）永远不能被修改或删除
- 只修改指定的section: {section_id}
- 遵循修改指导的语义方向
- 所有修改必须通过git patch格式
- 保持修改在字符限制内

输出格式：
SECTION_TO_MODIFY: {section_id}
OLD_CONTENT:
[原始内容]
NEW_CONTENT:
[修改后内容]
RATIONALE: [简要说明如何响应压力信号]"""

        return self.llm_fn(optimizer_prompt)
    
    def _map_pressure_to_guidance(self, pressure_type: str, direction: str,
                                  error_mode: str, magnitude: str) -> str:
        """Map structured pressure signals to modification guidance."""
        
        # Build guidance based on pressure type
        guidance_map = {
            'constraint_strictness': {
                'increase': '使评分标准更严格、更挑剔',
                'decrease': '使评分标准更宽松、更包容',
                'maintain': '保持当前严格度'
            },
            'evaluation_threshold': {
                'increase': '提高通过/优秀的门槛',
                'decrease': '降低通过/优秀的门槛',
                'maintain': '保持当前门槛'
            },
            'preference_weight': {
                'increase': '增加特定标准的权重',
                'decrease': '减少特定标准的权重',
                'maintain': '保持当前权重分配'
            },
            'ambiguity_tolerance': {
                'increase': '对模糊回答更宽容',
                'decrease': '对模糊回答更严格',
                'maintain': '保持对模糊性的当前态度'
            }
        }
        
        base_guidance = guidance_map.get(pressure_type, {}).get(direction, '适当调整')
        
        # Add error mode context
        error_context = {
            'overestimation': '（当前倾向于高估）',
            'underestimation': '（当前倾向于低估）',
            'variance': '（当前分数波动较大）'
        }
        
        context = error_context.get(error_mode, '')
        
        # Add magnitude hint
        magnitude_hints = {
            'strong': '进行明显调整',
            'medium': '进行适度调整',
            'weak': '进行微小调整'
        }
        
        magnitude_hint = magnitude_hints.get(magnitude, '')
        
        return f"{base_guidance}{context}。{magnitude_hint}。"
    
    def parse_modification(self, suggestion: str) -> Optional[Dict[str, str]]:
        """
        Parse modification suggestion into structured format.
        
        Args:
            suggestion: Text suggestion from LLM
            
        Returns:
            Dictionary with 'section', 'old_content', 'new_content', 'rationale'
            or None if parsing fails
        """
        lines = suggestion.strip().split('\n')
        
        result = {
            'section': '',
            'old_content': '',
            'new_content': '',
            'rationale': ''
        }
        
        current_field = None
        content_buffer = []
        
        for line in lines:
            if line.startswith('SECTION_TO_MODIFY:'):
                result['section'] = line.split(':', 1)[1].strip()
            elif line.startswith('OLD_CONTENT:'):
                if current_field and content_buffer:
                    result[current_field] = '\n'.join(content_buffer)
                current_field = 'old_content'
                content_buffer = []
            elif line.startswith('NEW_CONTENT:'):
                if current_field and content_buffer:
                    result[current_field] = '\n'.join(content_buffer)
                current_field = 'new_content'
                content_buffer = []
            elif line.startswith('RATIONALE:'):
                if current_field and content_buffer:
                    result[current_field] = '\n'.join(content_buffer)
                result['rationale'] = line.split(':', 1)[1].strip()
                current_field = None
            elif current_field:
                content_buffer.append(line)
        
        # Capture any remaining content
        if current_field and content_buffer:
            result[current_field] = '\n'.join(content_buffer)
        
        # Validate
        if not result['section'] or not result['new_content']:
            return None
        
        return result
    
    def validate_modification(self, modification: Dict[str, str],
                            learning_rate: float,
                            editable_sections: List[str],
                            meta_sections: List[str]) -> bool:
        """
        Validate that modification respects constraints.
        
        Args:
            modification: Parsed modification dictionary
            learning_rate: Current learning rate
            editable_sections: List of editable sections
            meta_sections: List of meta sections that cannot be modified
            
        Returns:
            True if modification is valid
        """
        section_name = modification['section']
        
        # Check section is not a meta section
        if section_name in meta_sections:
            return False
        
        # For new sections (not in editable_sections yet), they're allowed at high LR
        # For existing sections, they must be in editable_sections
        # This is checked implicitly - if it's not meta and optimizer suggests it, it's valid
        
        # Check character limit
        max_chars = self.compute_max_chars(learning_rate)
        old_len = len(modification.get('old_content', ''))
        new_len = len(modification['new_content'])
        char_diff = abs(new_len - old_len)
        
        if char_diff > max_chars:
            return False
        
        return True
    
    def generate_unified_diff(self, old_content: str, new_content: str,
                            section_name: str) -> str:
        """
        Generate unified diff for the modification.
        
        Args:
            old_content: Original section content
            new_content: Modified section content
            section_name: Name of the section
            
        Returns:
            Unified diff string
        """
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f'a/{section_name}',
            tofile=f'b/{section_name}',
            lineterm=''
        )
        
        return ''.join(diff)
