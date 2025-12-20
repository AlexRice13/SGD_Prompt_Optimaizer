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
    
    def generate_modification_suggestion(self, 
                                        current_prompt: str,
                                        proxy_gradient: str,
                                        learning_rate: float,
                                        editable_sections: List[str],
                                        meta_sections: List[str]) -> str:
        """
        Generate prompt modification suggestion.
        
        Args:
            current_prompt: Current JudgePrompt text
            proxy_gradient: Proxy gradient from GradientAgent
            learning_rate: Current learning rate
            editable_sections: List of sections that can be modified
            meta_sections: List of sections that cannot be modified or deleted
            
        Returns:
            Modification suggestion as text
        """
        permissions = self.get_permissions(learning_rate)
        max_chars = self.compute_max_chars(learning_rate)
        
        optimizer_prompt = f"""You are a prompt optimizer agent. Generate a modification suggestion for the Judge Prompt.

Current Judge Prompt:
{current_prompt}

Proxy Gradient (Analysis):
{proxy_gradient}

Optimization Constraints:
- Learning Rate: {learning_rate:.4f}
- Structural Edit Threshold: {self.structural_edit_threshold:.4f}
- Maximum character changes: {max_chars}
- Editable sections: {', '.join(editable_sections)}
- Meta sections (CANNOT modify or delete): {', '.join(meta_sections)}

Permissions:
- Modify content in editable sections: {permissions['modify_content']}
- Add new sections: {permissions['add_sections']} (only above threshold)
- Remove editable sections: {permissions['remove_sections']} (only above threshold)

Important Rules:
- Meta sections ({', '.join(meta_sections)}) can NEVER be modified or deleted
- At low LR: Only modify content of existing editable sections within character limit
- At high LR: Can also add new sections or remove existing editable sections
- All modifications must be via git patch format
- Do NOT change the overall scoring objective

Task: Generate a specific modification suggestion in unified diff format.
Focus on changes that address the issues identified in the proxy gradient.
Keep changes minimal and within the character limit.

Output format:
SECTION_TO_MODIFY: [section name]
OLD_CONTENT:
[original content]
NEW_CONTENT:
[modified content]
RATIONALE: [brief explanation]"""

        return self.llm_fn(optimizer_prompt)
    
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
