"""
Optimizer: Prompt modification with learning rate control.

Implements the Optimizer agent that directly modifies prompt sections
based on simple gradients, with modification strength controlled by learning rate.
"""

from typing import Callable, Dict, List, Optional
from prompts import OPTIMIZER_SIMPLE_PROMPT_TEMPLATE


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
                 debug: bool = False):
        """
        Initialize optimizer.
        
        Args:
            llm_fn: Function that takes a prompt and returns text
            initial_lr: Initial learning rate to compute modification strength
            debug: Enable full LLM output logging for debugging (default: False)
        
        Raises:
            ValueError: If initial_lr <= 0
        """
        if initial_lr <= 0:
            raise ValueError(f"initial_lr must be positive, got {initial_lr}")
        
        self.llm_fn = llm_fn
        self.initial_lr = initial_lr
        self.debug = debug
    
    def generate_modification_from_simple_gradient(self, 
                                                   current_prompt: str,
                                                   simple_gradient: Dict,
                                                   learning_rate: float,
                                                   editable_sections: List[str],
                                                   meta_sections: List[str]) -> str:
        """
        Generate prompt modification from simple gradient.
        
        Args:
            current_prompt: Current JudgePrompt text
            simple_gradient: Simple gradient dict with opti_direction and section_to_opti
            learning_rate: Current learning rate
            editable_sections: List of sections that can be modified
            meta_sections: List of sections that cannot be modified or deleted
            
        Returns:
            Modified section content as text
        """
        # Extract gradient fields
        opti_direction = simple_gradient.get('opti_direction', '')
        section_to_opti = simple_gradient.get('section_to_opti', '')
        
        if not opti_direction or not section_to_opti:
            print("Warning: Missing opti_direction or section_to_opti in gradient")
            return "NO_MODIFICATION"
        
        # Check if section is valid
        if section_to_opti in meta_sections:
            print(f"Warning: Cannot modify meta section '{section_to_opti}'")
            return "NO_MODIFICATION"
        
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
        
        # Use centralized simple prompt template
        from prompts import OPTIMIZER_SIMPLE_PROMPT_TEMPLATE
        optimizer_prompt = OPTIMIZER_SIMPLE_PROMPT_TEMPLATE.format(
            current_prompt=current_prompt,
            section_to_opti=section_to_opti,
            opti_direction=opti_direction,
            learning_rate=f"{learning_rate:.4f}",
            strength_desc=strength_desc,
            editable_sections=', '.join(editable_sections),
            meta_sections=', '.join(meta_sections)
        )

        llm_output = self.llm_fn(optimizer_prompt)
        
        # Log LLM output for debugging
        if self.debug:
            print(f"\n{'='*80}")
            print(f"=== FULL Optimizer LLM Output (Debug Mode) ===")
            print(f"Total length: {len(llm_output)} characters")
            print(f"\n{llm_output}")
            print(f"{'='*80}\n")
        else:
            print(f"\n=== Optimizer LLM Output ===")
            print(f"Total length: {len(llm_output)} characters")
            print(f"First 500 chars:\n{llm_output[:500]}")
            if len(llm_output) > 500:
                print(f"Last 500 chars:\n{llm_output[-500:]}")
            print("=" * 50)
        
        return llm_output
    
    def parse_modification(self, suggestion: str, section_name: str) -> Optional[str]:
        """
        Parse modification suggestion to extract new section content.
        
        In simplified version, the LLM output should be the complete new section content.
        
        Args:
            suggestion: Text suggestion from LLM (should be new section content)
            section_name: Name of section being modified
            
        Returns:
            New section content or None if empty
        """
        if not suggestion or suggestion.strip() == "NO_MODIFICATION":
            print("Parse: No modification suggested")
            return None
        
        # Clean up the output
        new_content = suggestion.strip()
        
        # Remove markdown code blocks if present
        import re
        code_block_match = re.search(r'```(?:text)?\s*\n(.*?)\n```', new_content, re.DOTALL)
        if code_block_match:
            new_content = code_block_match.group(1).strip()
        
        # Debug output
        print(f"\n=== Parse Result ===")
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
    
    def validate_modification(self, new_content: str,
                            section_name: str,
                            learning_rate: float,
                            editable_sections: List[str],
                            meta_sections: List[str]) -> bool:
        """
        Validate that modification respects constraints.
        
        Args:
            new_content: New section content
            section_name: Name of section being modified
            learning_rate: Current learning rate
            editable_sections: List of editable sections
            meta_sections: List of sections that cannot be modified
            
        Returns:
            True if modification is valid
        """
        # Check section is not a meta section
        if section_name in meta_sections:
            print(f"Validation failed: {section_name} is a meta section")
            return False
        
        # Check new_content is not empty
        if not new_content or not new_content.strip():
            print("Validation failed: Empty new content")
            return False
        
        return True
