"""
JudgePrompt: Structured prompt representation with meta sections.

Represents the JudgePrompt as a trainable parameter with constraints.
"""

from typing import Dict, List, Optional, Set
import json


class JudgePrompt:
    """
    Structured prompt representation with sections.
    
    Sections can be marked as meta (frozen) or editable, allowing controlled
    modifications during the optimization process.
    
    - meta_sections: Sections that CANNOT be modified or deleted at any stage
    - All other sections are automatically editable with LR-based permissions
    """
    
    def __init__(self, sections: Dict[str, str], meta_sections: List[str]):
        """
        Initialize JudgePrompt with sections.
        
        Args:
            sections: Dictionary of section_name -> section_content
            meta_sections: List of section names that cannot be modified or deleted
                          All sections NOT in this list are automatically editable
        """
        self.sections = sections.copy()
        self.meta_sections = set(meta_sections)
        self._validate()
    
    def _validate(self):
        """Validate that meta sections exist in sections."""
        for section in self.meta_sections:
            if section not in self.sections:
                raise ValueError(f"Meta section '{section}' not found in sections")
    
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
    
    def get_editable_sections(self) -> Set[str]:
        """
        Get list of editable sections (all sections except meta_sections).
        
        Returns:
            Set of editable section names
        """
        return set(self.sections.keys()) - self.meta_sections
    
    def update_section(self, section_name: str, new_content: str) -> bool:
        """
        Update a section's content if it's not a meta section.
        
        Args:
            section_name: Name of the section to update
            new_content: New content for the section
            
        Returns:
            True if update was successful, False if section is meta
        """
        if section_name in self.meta_sections:
            return False
        
        self.sections[section_name] = new_content
        return True
    
    def add_section(self, section_name: str, content: str) -> bool:
        """
        Add a new section (only allowed if not a meta section).
        
        Args:
            section_name: Name of the new section
            content: Content for the new section
            
        Returns:
            True if added successfully, False if name conflicts with meta section
        """
        if section_name in self.meta_sections:
            return False
        
        self.sections[section_name] = content
        return True
    
    def remove_section(self, section_name: str) -> bool:
        """
        Remove a section (only allowed if not a meta section).
        
        Args:
            section_name: Name of the section to remove
            
        Returns:
            True if removed successfully, False if section is meta or doesn't exist
        """
        if section_name in self.meta_sections or section_name not in self.sections:
            return False
        
        del self.sections[section_name]
        return True
    
    def get_section(self, section_name: str) -> Optional[str]:
        """Get content of a specific section."""
        return self.sections.get(section_name)
    
    def is_meta(self, section_name: str) -> bool:
        """Check if a section is a meta section (frozen)."""
        return section_name in self.meta_sections
    
    def is_editable(self, section_name: str) -> bool:
        """Check if a section is editable (not a meta section)."""
        return section_name in self.sections and section_name not in self.meta_sections
    
    def to_dict(self) -> dict:
        """Serialize to dictionary.
        
        Returns a dictionary with a shallow copy of sections to prevent
        external code from modifying the internal state. This is sufficient
        since section values are immutable strings.
        """
        return {
            "sections": self.sections.copy(),
            "meta_sections": list(self.meta_sections)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'JudgePrompt':
        """Deserialize from dictionary."""
        # Support both old format (editable_sections) and new format (meta_sections)
        if "meta_sections" in data:
            return cls(data["sections"], data["meta_sections"])
        elif "editable_sections" in data:
            # Convert old format: invert to meta_sections
            all_sections = set(data["sections"].keys())
            editable = set(data["editable_sections"])
            meta = list(all_sections - editable)
            return cls(data["sections"], meta)
        else:
            raise ValueError("JSON must contain either 'meta_sections' or 'editable_sections'")
    
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


if __name__ == '__main__':
    """Unit tests for JudgePrompt class."""
    import tempfile
    import os
    
    print("Running JudgePrompt unit tests...")
    
    # Test 1: Basic initialization
    print("\n1. Testing basic initialization...")
    sections = {'Criteria': 'Test criteria', 'Scale': '1-10', 'Format': 'Output score'}
    meta_sections = ['Scale', 'Format']
    prompt = JudgePrompt(sections, meta_sections)
    assert len(prompt.sections) == 3
    assert len(prompt.meta_sections) == 2
    assert 'Criteria' in prompt.get_editable_sections()
    print("   ✓ Basic initialization works")
    
    # Test 2: Update section
    print("\n2. Testing section updates...")
    assert prompt.update_section('Criteria', 'New criteria') == True
    assert prompt.get_section('Criteria') == 'New criteria'
    assert prompt.update_section('Scale', 'New scale') == False  # Meta section
    assert prompt.get_section('Scale') == '1-10'  # Unchanged
    print("   ✓ Section updates work correctly")
    
    # Test 3: Add/remove sections
    print("\n3. Testing add/remove sections...")
    assert prompt.add_section('NewSection', 'New content') == True
    assert 'NewSection' in prompt.sections
    assert prompt.add_section('Scale', 'Conflict') == False  # Meta section
    assert prompt.remove_section('NewSection') == True
    assert 'NewSection' not in prompt.sections
    assert prompt.remove_section('Scale') == False  # Meta section
    print("   ✓ Add/remove sections work correctly")
    
    # Test 4: to_dict returns copy (bug fix)
    print("\n4. Testing to_dict returns copy...")
    d = prompt.to_dict()
    d['sections']['Criteria'] = 'HACKED'
    assert prompt.get_section('Criteria') == 'New criteria'  # Not affected
    print("   ✓ to_dict returns independent copy")
    
    # Test 5: from_dict/to_dict round-trip
    print("\n5. Testing serialization round-trip...")
    prompt2 = JudgePrompt.from_dict(prompt.to_dict())
    assert prompt2.get_section('Criteria') == prompt.get_section('Criteria')
    assert prompt2.meta_sections == prompt.meta_sections
    prompt.update_section('Criteria', 'Modified')
    assert prompt2.get_section('Criteria') == 'New criteria'  # Independent
    print("   ✓ Serialization round-trip works")
    
    # Test 6: Save/load from file
    print("\n6. Testing save/load...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    try:
        prompt.save(filename)
        loaded = JudgePrompt.load(filename)
        assert loaded.get_section('Criteria') == 'Modified'
        assert loaded.meta_sections == prompt.meta_sections
        print("   ✓ Save/load works correctly")
    finally:
        os.unlink(filename)
    
    # Test 7: get_full_prompt
    print("\n7. Testing get_full_prompt...")
    full = prompt.get_full_prompt()
    assert '## Criteria' in full
    assert '## Scale' in full
    assert 'Modified' in full
    print("   ✓ get_full_prompt works correctly")
    
    # Test 8: Old format compatibility
    print("\n8. Testing backward compatibility...")
    old_format = {
        'sections': {'S1': 'Content1', 'S2': 'Content2'},
        'editable_sections': ['S1']
    }
    prompt3 = JudgePrompt.from_dict(old_format)
    assert 'S1' in prompt3.get_editable_sections()
    assert 'S2' in prompt3.meta_sections
    print("   ✓ Old format compatibility works")
    
    print("\n" + "="*50)
    print("All JudgePrompt tests passed! ✓")
    print("="*50)
