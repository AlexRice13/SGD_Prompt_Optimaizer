"""
Optimizer: Prompt modification with learning rate control.

Implements the Optimizer agent that directly modifies prompt sections
based on simple gradients, with modification strength controlled by learning rate.
"""

from typing import Callable, Dict, List, Optional
from prompts import (
    OPTIMIZER_SIMPLE_PROMPT_TEMPLATE,
    SIMPLIFICATION_HINT_MANY_SECTIONS,
    SIMPLIFICATION_HINT_ADD_SECTION,
    SIMPLIFICATION_HINT_DEFAULT
)
from constants import STRUCTURAL_EDIT_LR_THRESHOLD, can_perform_structural_edit


class PromptOptimizer:
    """
    Optimizer that generates prompt modifications based on simple gradients.
    
    Takes simple gradient (opti_direction + section_to_opti) and directly 
    modifies the section content. Modification strength is controlled by 
    learning rate through prompt engineering.
    
    Note: meta_sections can NEVER be modified at any LR stage.
    """
    
    def __init__(self, llm_fn: Callable[[str], str], 
                 initial_lr: float = 0.1,
                 debug: bool = False,
                 simplification_threshold: int = 5):
        """
        Initialize optimizer.
        
        Args:
            llm_fn: Function that takes a prompt and returns text
            initial_lr: Initial learning rate to compute modification strength
            debug: Enable full LLM output logging for debugging (default: False)
            simplification_threshold: Max recommended section count before simplification hints (default: 5)
        
        Raises:
            ValueError: If initial_lr <= 0
        """
        if initial_lr <= 0:
            raise ValueError(f"initial_lr must be positive, got {initial_lr}")
        
        self.llm_fn = llm_fn
        self.initial_lr = initial_lr
        self.debug = debug
        self.simplification_threshold = simplification_threshold
    
    def generate_modification_from_gradient(self, 
                                           current_prompt: str,
                                           modification: Dict,
                                           learning_rate: float,
                                           editable_sections: List[str],
                                           meta_sections: List[str]) -> Dict:
        """
        Generate prompt modification from a single gradient modification.
        
        Args:
            current_prompt: Current JudgePrompt text
            modification: Single modification dict with action, section_name, and opti_direction
            learning_rate: Current learning rate
            editable_sections: List of sections that can be modified
            meta_sections: List of sections that cannot be modified or deleted
            
        Returns:
            Dict with action, section_name, and content (for edit/add) or success status
        """
        action = modification.get('action', '')
        section_name = modification.get('section_name', '')
        opti_direction = modification.get('opti_direction', '')
        
        if not action or not section_name:
            print("Warning: Missing action or section_name in modification")
            return {'action': 'skip', 'section_name': section_name, 'reason': 'missing_fields'}
        
        # Validate action type
        if action not in ['edit', 'add', 'remove']:
            print(f"Warning: Invalid action '{action}'")
            return {'action': 'skip', 'section_name': section_name, 'reason': 'invalid_action'}
        
        # Check LR threshold for structural changes
        if action in ['add', 'remove'] and not can_perform_structural_edit(learning_rate):
            print(f"Warning: Cannot {action} section at LR={learning_rate:.4f} < {STRUCTURAL_EDIT_LR_THRESHOLD}")
            return {'action': 'skip', 'section_name': section_name, 'reason': 'lr_threshold'}
        
        # Check meta section constraints
        if section_name in meta_sections:
            print(f"Warning: Cannot perform {action} on meta section '{section_name}'")
            return {'action': 'skip', 'section_name': section_name, 'reason': 'meta_section'}
        
        # For remove action, no LLM call needed
        if action == 'remove':
            return {
                'action': 'remove',
                'section_name': section_name,
                'content': None
            }
        
        # For edit/add actions, need opti_direction
        if not opti_direction:
            print(f"Warning: Missing opti_direction for {action} action")
            return {'action': 'skip', 'section_name': section_name, 'reason': 'missing_direction'}
        
        # Calculate modification strength based on learning rate
        lr_ratio = learning_rate / self.initial_lr
        lr_ratio = max(0.0, min(1.0, lr_ratio))
        
        # Map learning rate to modification strength description
        if lr_ratio > 0.7:
            strength_desc = "大幅度"
        elif lr_ratio > 0.3:
            strength_desc = "适度"
        else:
            strength_desc = "轻微"
        
        # Prepare task description based on action
        if action == 'edit':
            task_description = f"请直接输出修改后的'{section_name}' section的完整内容。不要使用git patch格式或其他复杂格式。"
            output_description = "修改后的section内容"
        else:  # action == 'add'
            task_description = f"请直接输出新增的'{section_name}' section的完整内容。这是一个新section，需要根据优化方向创建。"
            output_description = "新section的内容"
        
        # Calculate total sections and generate simplification hint using centralized templates
        total_sections = len(editable_sections) + len(meta_sections)
        
        # Simplification pressure: Occam's Razor principle
        if total_sections > self.simplification_threshold:
            simplification_hint = SIMPLIFICATION_HINT_MANY_SECTIONS.format(
                total_sections=total_sections,
                threshold=self.simplification_threshold
            )
        elif action == 'add':
            simplification_hint = SIMPLIFICATION_HINT_ADD_SECTION
        else:
            simplification_hint = SIMPLIFICATION_HINT_DEFAULT
        
        # Use centralized prompt template with dynamic fields
        from prompts import OPTIMIZER_SIMPLE_PROMPT_TEMPLATE
        optimizer_prompt = OPTIMIZER_SIMPLE_PROMPT_TEMPLATE.format(
            current_prompt=current_prompt,
            action=action,
            section_name=section_name,
            opti_direction=opti_direction,
            learning_rate=f"{learning_rate:.4f}",
            strength_desc=strength_desc,
            editable_sections=', '.join(editable_sections),
            meta_sections=', '.join(meta_sections),
            total_sections=total_sections,
            simplification_hint=simplification_hint,
            task_description=task_description,
            output_description=output_description
        )

        llm_output = self.llm_fn(optimizer_prompt)
        
        # Log LLM output for debugging
        if self.debug:
            print(f"\n{'='*80}")
            print(f"=== FULL Optimizer LLM Output (Debug Mode) - {action} {section_name} ===")
            print(f"Total length: {len(llm_output)} characters")
            print(f"\n{llm_output}")
            print(f"{'='*80}\n")
        else:
            print(f"\n=== Optimizer LLM Output - {action} {section_name} ===")
            print(f"Total length: {len(llm_output)} characters")
            print(f"First 500 chars:\n{llm_output[:500]}")
            if len(llm_output) > 500:
                print(f"Last 500 chars:\n{llm_output[-500:]}")
            print("=" * 50)
        
        return {
            'action': action,
            'section_name': section_name,
            'content': llm_output
        }
    
    def parse_modification(self, result: Dict) -> Optional[str]:
        """
        Parse modification result to extract new section content.
        
        Args:
            result: Result dict from generate_modification_from_gradient
            
        Returns:
            New section content or None if not applicable
        """
        action = result.get('action', '')
        section_name = result.get('section_name', '')
        content = result.get('content', '')
        
        if action == 'skip':
            print(f"Parse: Skipping modification due to {result.get('reason', 'unknown')}")
            return None
        
        if action == 'remove':
            # For remove action, no content to parse
            return None
        
        if not content or content.strip() == "NO_MODIFICATION":
            print("Parse: No modification suggested")
            return None
        
        # Clean up the output
        new_content = content.strip()
        
        # Remove markdown code blocks if present
        import re
        code_block_match = re.search(r'```(?:text)?\s*\n(.*?)\n```', new_content, re.DOTALL)
        if code_block_match:
            new_content = code_block_match.group(1).strip()
        
        # Debug output
        print(f"\n=== Parse Result ===")
        print(f"Action: '{action}'")
        print(f"Section: '{section_name}'")
        print(f"New content length: {len(new_content)}")
        if new_content:
            print(f"New content preview: {new_content[:200]}...")
        else:
            print("New content: EMPTY!")
        print("=" * 50)
        
        # Validate - need non-empty content
        if not new_content:
            print("Parse failed: No new content provided")
            return None
        
        return new_content
    
    def validate_modification(self, result: Dict,
                            learning_rate: float,
                            editable_sections: List[str],
                            meta_sections: List[str]) -> bool:
        """
        Validate that modification respects constraints.
        
        Args:
            result: Result dict from generate_modification_from_gradient
            learning_rate: Current learning rate
            editable_sections: List of editable sections
            meta_sections: List of sections that cannot be modified
            
        Returns:
            True if modification is valid
        """
        action = result.get('action', '')
        section_name = result.get('section_name', '')
        content = result.get('content', '')
        
        # Skip actions are not valid for application
        if action == 'skip':
            print(f"Validation failed: Action is skip due to {result.get('reason', 'unknown')}")
            return False
        
        # Check section is not a meta section
        if section_name in meta_sections:
            print(f"Validation failed: {section_name} is a meta section")
            return False
        
        # Check LR threshold for structural operations
        if action in ['add', 'remove'] and not can_perform_structural_edit(learning_rate):
            print(f"Validation failed: {action} not allowed at LR={learning_rate:.4f} < {STRUCTURAL_EDIT_LR_THRESHOLD}")
            return False
        
        # For edit/add actions, check content is not empty
        if action in ['edit', 'add']:
            if not content or not content.strip():
                print("Validation failed: Empty content for edit/add action")
                return False
        
        return True


if __name__ == '__main__':
    """Unit tests for PromptOptimizer class."""
    
    print("Running PromptOptimizer unit tests...")
    
    # Test 1: Basic initialization
    print("\n1. Testing basic initialization...")
    
    def mock_llm_fn(prompt):
        """Mock LLM function."""
        return "Mock optimizer output: new content here"
    
    optimizer = PromptOptimizer(mock_llm_fn, initial_lr=0.1)
    assert optimizer.initial_lr == 0.1
    assert optimizer.llm_fn == mock_llm_fn
    print("   ✓ Initialization works")
    
    # Test 2: Validation - skip action
    print("\n2. Testing validation - skip action...")
    result_skip = {'action': 'skip', 'section_name': 'Test', 'reason': 'test'}
    assert optimizer.validate_modification(result_skip, 0.1, ['S1'], ['M1']) == False
    print("   ✓ Skip action validation works")
    
    # Test 3: Validation - meta section
    print("\n3. Testing validation - meta section protection...")
    result_meta = {'action': 'edit', 'section_name': 'MetaSection', 'content': 'New'}
    assert optimizer.validate_modification(result_meta, 0.1, ['S1'], ['MetaSection']) == False
    print("   ✓ Meta section protection works")
    
    # Test 4: Validation - empty content
    print("\n4. Testing validation - empty content...")
    result_empty = {'action': 'edit', 'section_name': 'S1', 'content': ''}
    assert optimizer.validate_modification(result_empty, 0.1, ['S1'], []) == False
    print("   ✓ Empty content validation works")
    
    # Test 5: Validation - valid edit
    print("\n5. Testing validation - valid modification...")
    result_valid = {'action': 'edit', 'section_name': 'S1', 'content': 'Valid content'}
    assert optimizer.validate_modification(result_valid, 0.1, ['S1'], []) == True
    print("   ✓ Valid modification passes")
    
    # Test 6: Parse modification
    print("\n6. Testing parse_modification...")
    result_with_content = {
        'action': 'edit',
        'section_name': 'S1',
        'content': 'Test content'
    }
    parsed = optimizer.parse_modification(result_with_content)
    assert parsed == 'Test content'
    print("   ✓ Parse modification works")
    
    # Test 7: Parse with NO_MODIFICATION
    print("\n7. Testing parse with NO_MODIFICATION...")
    result_no_mod = {
        'action': 'edit',
        'section_name': 'S1',
        'content': 'NO_MODIFICATION'
    }
    parsed_none = optimizer.parse_modification(result_no_mod)
    assert parsed_none is None
    print("   ✓ NO_MODIFICATION handled correctly")
    
    print("\n" + "="*50)
    print("All PromptOptimizer tests passed! ✓")
    print("="*50)
