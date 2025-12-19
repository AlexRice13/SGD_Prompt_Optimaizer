"""
Version Control: Git-based logging for prompt evolution.

Implements version control for tracking prompt evolution using git commits.
Each optimization step is recorded as a commit with metadata.
"""

import subprocess
import json
from typing import Dict, Optional
from pathlib import Path


class VersionControl:
    """
    Git-based version control for prompt evolution.
    
    Each prompt update is a git commit, creating a full history
    of the optimization process.
    """
    
    def __init__(self, repo_path: str, prompt_filename: str = "judge_prompt.json"):
        """
        Initialize version control.
        
        Args:
            repo_path: Path to git repository
            prompt_filename: Filename for judge prompt
        """
        self.repo_path = Path(repo_path)
        self.prompt_filename = prompt_filename
        self.prompt_path = self.repo_path / prompt_filename
        
        # Initialize git repo if needed
        self._init_repo()
    
    def _init_repo(self):
        """Initialize git repository if it doesn't exist."""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            subprocess.run(
                ["git", "init"],
                cwd=self.repo_path,
                capture_output=True
            )
    
    def commit_prompt_update(self, 
                           step: int,
                           train_loss: float,
                           val_loss: float,
                           metrics: Dict[str, float],
                           gradient_summary: str,
                           modification_rationale: str = "") -> bool:
        """
        Commit prompt update with metadata.
        
        Args:
            step: Optimization step number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Dictionary of metrics
            gradient_summary: Summary of proxy gradient
            modification_rationale: Rationale for modification
            
        Returns:
            True if commit was successful
        """
        # Create commit message with metadata
        commit_msg = self._create_commit_message(
            step, train_loss, val_loss, metrics,
            gradient_summary, modification_rationale
        )
        
        # Stage prompt file
        try:
            subprocess.run(
                ["git", "add", self.prompt_filename],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            
            # Commit
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _create_commit_message(self,
                              step: int,
                              train_loss: float,
                              val_loss: float,
                              metrics: Dict[str, float],
                              gradient_summary: str,
                              modification_rationale: str) -> str:
        """Create structured commit message."""
        lines = [
            f"Step {step}: Prompt optimization update",
            "",
            f"Train Loss: {train_loss:.4f}",
            f"Val Loss: {val_loss:.4f}",
            "",
            "Metrics:",
        ]
        
        for key, value in metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        
        lines.append("")
        lines.append("Gradient Summary:")
        lines.append(gradient_summary[:200] + "..." if len(gradient_summary) > 200 else gradient_summary)
        
        if modification_rationale:
            lines.append("")
            lines.append("Modification Rationale:")
            lines.append(modification_rationale[:200] + "..." if len(modification_rationale) > 200 else modification_rationale)
        
        return "\n".join(lines)
    
    def create_checkpoint_tag(self, step: int, is_best: bool = False):
        """
        Create a git tag for checkpoint.
        
        Args:
            step: Step number
            is_best: Whether this is the best checkpoint
        """
        tag_name = f"best-checkpoint" if is_best else f"checkpoint-step-{step}"
        
        try:
            subprocess.run(
                ["git", "tag", "-f", tag_name],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            pass
    
    def checkout_checkpoint(self, step: Optional[int] = None, tag: Optional[str] = None):
        """
        Checkout a specific checkpoint.
        
        Args:
            step: Step number to checkout
            tag: Tag name to checkout (overrides step)
        """
        if tag:
            target = tag
        elif step is not None:
            target = f"checkpoint-step-{step}"
        else:
            target = "best-checkpoint"
        
        try:
            subprocess.run(
                ["git", "checkout", target],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            pass
    
    def get_commit_history(self, max_count: int = 10) -> list:
        """
        Get commit history.
        
        Args:
            max_count: Maximum number of commits to retrieve
            
        Returns:
            List of commit information dictionaries
        """
        try:
            result = subprocess.run(
                ["git", "log", f"--max-count={max_count}", 
                 "--pretty=format:%H|%s|%ai"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    hash_val, subject, date = line.split('|', 2)
                    commits.append({
                        'hash': hash_val,
                        'subject': subject,
                        'date': date
                    })
            
            return commits
        except subprocess.CalledProcessError:
            return []
