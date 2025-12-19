"""
JudgePrompt: Structured prompt representation with editable sections.

Represents the JudgePrompt as a trainable parameter with constraints.
"""

from typing import Dict, List, Optional
import json


class JudgePrompt:
    """
    Structured prompt representation with sections.
    
    Sections can be marked as editable or frozen, allowing controlled
    modifications during the optimization process.
    """
    
    def __init__(self, sections: Dict[str, str], editable_sections: List[str]):
        """
        Initialize JudgePrompt with sections.
        
        Args:
            sections: Dictionary of section_name -> section_content
            editable_sections: List of section names that can be modified
        """
        self.sections = sections.copy()
        self.editable_sections = set(editable_sections)
        self._validate()
    
    def _validate(self):
        """Validate that editable sections exist in sections."""
        for section in self.editable_sections:
            if section not in self.sections:
                raise ValueError(f"Editable section '{section}' not found in sections")
    
    def get_full_prompt(self) -> str:
        """
        Assemble the complete prompt from all sections.
        
        Returns:
            Complete prompt text
        """
        parts = []
        for section_name, content in self.sections.items():
            parts.append(f"## {section_name}")
            parts.append(content)
            parts.append("")
        return "\n".join(parts)
    
    def update_section(self, section_name: str, new_content: str) -> bool:
        """
        Update a section's content if it's editable.
        
        Args:
            section_name: Name of the section to update
            new_content: New content for the section
            
        Returns:
            True if update was successful, False otherwise
        """
        if section_name not in self.editable_sections:
            return False
        
        self.sections[section_name] = new_content
        return True
    
    def get_section(self, section_name: str) -> Optional[str]:
        """Get content of a specific section."""
        return self.sections.get(section_name)
    
    def is_editable(self, section_name: str) -> bool:
        """Check if a section is editable."""
        return section_name in self.editable_sections
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "sections": self.sections,
            "editable_sections": list(self.editable_sections)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'JudgePrompt':
        """Deserialize from dictionary."""
        return cls(data["sections"], data["editable_sections"])
    
    def save(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'JudgePrompt':
        """Load from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        return self.get_full_prompt()
