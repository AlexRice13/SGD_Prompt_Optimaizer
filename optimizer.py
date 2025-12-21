"""
Optimizer: Prompt modification with learning rate-based permissions.

Implements the Optimizer agent that generates prompt modification suggestions
based on proxy gradients, with permissions bound to current learning rate.
"""

from typing import Callable, Dict, List, Optional
import difflib
from prompts import OPTIMIZER_PROMPT_TEMPLATE


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
                 initial_lr: float = 0.1,
                 base_char_limit: int = 300,
                 debug: bool = False):
        """
        Initialize optimizer.
        
        Args:
            llm_fn: Function that takes a prompt and returns text
            structural_edit_threshold_ratio: Ratio of max LR above which structural 
                                             edits (add/remove sections) are allowed.
                                             E.g., 0.5 means structural edits only when
                                             current_lr >= 0.5 * initial_lr
            initial_lr: Initial learning rate to compute threshold
            base_char_limit: Base character limit at initial LR (default: 300)
            debug: Enable full LLM output logging for debugging (default: False)
        
        Raises:
            ValueError: If initial_lr <= 0
        """
        if initial_lr <= 0:
            raise ValueError(f"initial_lr must be positive, got {initial_lr}")
        
        self.llm_fn = llm_fn
        self.structural_edit_threshold_ratio = structural_edit_threshold_ratio
        self.initial_lr = initial_lr
        self.base_char_limit = base_char_limit
        self.debug = debug
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
    
    def compute_max_chars(self, learning_rate: float) -> int:
        """
        Compute maximum character changes based on learning rate ratio.
        
        Character limit scales with (current_lr / initial_lr) ratio,
        so modifications become more constrained as training progresses.
        Note: LR ratio is clamped to [0, 1] range, so if LR exceeds initial_lr
        (which can happen in some schedulers), the char limit is capped at base_char_limit.
        
        Args:
            learning_rate: Current learning rate
            
        Returns:
            Maximum allowed character changes
        """
        # Scale character limit with LR ratio relative to initial LR
        lr_ratio = learning_rate / self.initial_lr
        lr_ratio = max(0.0, min(1.0, lr_ratio))  # Clamp to [0, 1]
        
        return max(10, int(self.base_char_limit * lr_ratio))
    
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
        
        # Check consistency - convert to bool for comparison to handle string "true"/"false"
        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes')
            return bool(val)
        
        ack_add = to_bool(ack.get('allow_add_section', False))
        ack_remove = to_bool(ack.get('allow_delete_section', False))
        
        if ack_add != permissions['add_sections']:
            print(f"Warning: Gradient action space mismatch on add_sections (ack={ack_add}, perm={permissions['add_sections']})")
        if ack_remove != permissions['remove_sections']:
            print(f"Warning: Gradient action space mismatch on remove_sections (ack={ack_remove}, perm={permissions['remove_sections']})")
        
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
        
        # Use centralized prompt template
        optimizer_prompt = OPTIMIZER_PROMPT_TEMPLATE.format(
            current_prompt=current_prompt,
            section_id=section_id,
            pressure_type=pressure_type,
            direction=direction,
            error_mode=error_mode,
            magnitude=magnitude,
            modification_guidance=modification_guidance,
            learning_rate=f"{learning_rate:.4f}",
            max_chars=max_chars,
            editable_sections=', '.join(editable_sections),
            meta_sections=', '.join(meta_sections),
            modify_content=permissions['modify_content'],
            add_sections=permissions['add_sections'],
            remove_sections=permissions['remove_sections'],
            meta_sections_list=', '.join(meta_sections)
        )

        llm_output = self.llm_fn(optimizer_prompt)
        
        # Log LLM output for debugging
        if self.debug:
            # Full output when debug mode is enabled
            print(f"\n{'='*80}")
            print(f"=== FULL Optimizer LLM Output (Debug Mode) ===")
            print(f"Total length: {len(llm_output)} characters")
            print(f"\n{llm_output}")
            print(f"{'='*80}\n")
        else:
            # Abbreviated output for normal mode
            print(f"\n=== Optimizer LLM Output ===")
            print(f"Total length: {len(llm_output)} characters")
            print(f"First 500 chars:\n{llm_output[:500]}")
            print(f"Last 500 chars:\n{llm_output[-500:]}")
            print("=" * 50)
        
        return llm_output
    
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
        
        Uses regex for robust field extraction to handle large content blocks.
        
        Args:
            suggestion: Text suggestion from LLM
            
        Returns:
            Dictionary with 'section', 'old_content', 'new_content', 'rationale'
            or None if parsing fails
        """
        if not suggestion or suggestion.strip() == "NO_MODIFICATION":
            print("Parse: No modification suggested")
            return None
        
        import re
        
        # Try to extract from markdown code blocks first
        code_block_match = re.search(r'```(?:text)?\s*\n(.*?)\n```', suggestion, re.DOTALL)
        if code_block_match:
            suggestion = code_block_match.group(1)
        
        result = {
            'section': '',
            'old_content': '',
            'new_content': '',
            'rationale': ''
        }
        
        # Extract section name (case-insensitive)
        section_match = re.search(r'(?:SECTION_TO_MODIFY|SECTION):\s*(.+?)(?:\n|$)', suggestion, re.IGNORECASE)
        if section_match:
            result['section'] = section_match.group(1).strip()
        
        # Extract OLD_CONTENT (everything between OLD_CONTENT: and NEW_CONTENT:)
        old_content_match = re.search(
            r'(?:OLD_CONTENT|OLD):\s*(.*?)(?=(?:NEW_CONTENT|NEW|RATIONALE|REASON):|$)', 
            suggestion, 
            re.IGNORECASE | re.DOTALL
        )
        if old_content_match:
            result['old_content'] = old_content_match.group(1).strip()
        
        # Extract NEW_CONTENT (everything between NEW_CONTENT: and RATIONALE:)
        new_content_match = re.search(
            r'(?:NEW_CONTENT|NEW):\s*(.*?)(?=(?:RATIONALE|REASON):|$)', 
            suggestion, 
            re.IGNORECASE | re.DOTALL
        )
        if new_content_match:
            result['new_content'] = new_content_match.group(1).strip()
        
        # Extract RATIONALE
        rationale_match = re.search(r'(?:RATIONALE|REASON):\s*(.+?)$', suggestion, re.IGNORECASE | re.DOTALL)
        if rationale_match:
            result['rationale'] = rationale_match.group(1).strip()
        
        # Debug output
        print(f"\n=== Parse Result ===")
        print(f"Section: '{result['section']}'")
        print(f"Old content length: {len(result['old_content'])}")
        print(f"New content length: {len(result['new_content'])}")
        if result['new_content']:
            print(f"New content preview: {result['new_content'][:200]}...")
        else:
            print("New content: EMPTY!")
        print(f"Rationale: '{result['rationale']}'")
        print("=" * 50)
        
        # Validate - only need section and new_content
        if not result['section']:
            print("Parse failed: No section specified")
            return None
        if not result['new_content']:
            print("Parse failed: No new content provided")
            print("This usually means:")
            print("1. LLM didn't output NEW_CONTENT field")
            print("2. LLM output was truncated")
            print("3. NEW_CONTENT field has a typo")
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
