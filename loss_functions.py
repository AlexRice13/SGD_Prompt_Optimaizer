"""
Loss Functions: MAE and Ranking Loss implementations.

Implements both absolute value alignment (MAE) and distribution/ranking
alignment (pairwise ranking loss) as described in the framework.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import kendalltau, spearmanr


class LossFunctions:
    """
    Combined loss function for prompt optimization.
    
    L = α · MAE + β · RankLoss
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize loss functions with weights.
        
        Args:
            alpha: Weight for MAE (absolute value alignment)
            beta: Weight for ranking loss (distribution/ranking alignment)
        """
        self.alpha = alpha
        self.beta = beta
    
    @staticmethod
    def mae_loss(judge_scores: np.ndarray, human_scores: np.ndarray) -> float:
        """
        Mean Absolute Error loss.
        
        Args:
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            
        Returns:
            MAE loss value
        """
        return float(np.mean(np.abs(judge_scores - human_scores)))
    
    @staticmethod
    def huber_loss(judge_scores: np.ndarray, human_scores: np.ndarray, 
                   delta: float = 1.0) -> float:
        """
        Huber loss (robust alternative to MAE).
        
        Args:
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            delta: Threshold for quadratic vs linear loss
            
        Returns:
            Huber loss value
        """
        errors = np.abs(judge_scores - human_scores)
        quadratic = errors <= delta
        
        loss = np.where(
            quadratic,
            0.5 * errors ** 2,
            delta * (errors - 0.5 * delta)
        )
        
        return float(np.mean(loss))
    
    @staticmethod
    def pairwise_ranking_loss(judge_scores: np.ndarray, 
                             human_scores: np.ndarray,
                             margin: float = 0.0) -> float:
        """
        Pairwise ranking loss with margin.
        
        Penalizes when the relative ordering between pairs disagrees
        with human scores.
        
        Args:
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            margin: Margin for ranking loss
            
        Returns:
            Pairwise ranking loss value
        """
        n = len(judge_scores)
        if n < 2:
            return 0.0
        
        total_loss = 0.0
        pair_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # Human preference
                human_diff = human_scores[i] - human_scores[j]
                # Judge preference
                judge_diff = judge_scores[i] - judge_scores[j]
                
                # If human prefers i over j (human_diff > 0)
                # Judge should also prefer i over j (judge_diff > 0)
                if abs(human_diff) > 1e-6:  # Only consider pairs with clear preference
                    sign_mismatch = np.sign(human_diff) != np.sign(judge_diff)
                    if sign_mismatch:
                        total_loss += abs(human_diff)
                    else:
                        # Even if signs match, penalize if margin not satisfied
                        if abs(judge_diff) < margin:
                            total_loss += (margin - abs(judge_diff))
                    
                    pair_count += 1
        
        return total_loss / max(pair_count, 1)
    
    @staticmethod
    def kendall_tau_loss(judge_scores: np.ndarray, human_scores: np.ndarray) -> float:
        """
        Kendall Tau based loss (1 - tau).
        
        Returns a value between 0 and 2, where 0 means perfect correlation.
        """
        if len(judge_scores) < 2:
            return 0.0
        
        tau, _ = kendalltau(judge_scores, human_scores)
        # Convert correlation [-1, 1] to loss [0, 2]
        # Handle NaN case (identical arrays or other edge cases)
        return 1.0 - tau if not np.isnan(tau) else 1.0
    
    @staticmethod
    def spearman_loss(judge_scores: np.ndarray, human_scores: np.ndarray) -> float:
        """
        Spearman correlation based loss (1 - rho).
        
        Returns a value between 0 and 2, where 0 means perfect correlation.
        """
        if len(judge_scores) < 2:
            return 0.0
        
        rho, _ = spearmanr(judge_scores, human_scores)
        # Convert correlation [-1, 1] to loss [0, 2]
        # Handle NaN case (identical arrays or other edge cases)
        return 1.0 - rho if not np.isnan(rho) else 1.0
    
    def compute_loss(self, judge_scores: np.ndarray, human_scores: np.ndarray,
                    loss_type: str = 'pairwise') -> Tuple[float, float, float]:
        """
        Compute combined loss.
        
        Args:
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            loss_type: Type of ranking loss ('pairwise', 'kendall', 'spearman')
            
        Returns:
            Tuple of (total_loss, mae_component, rank_component)
        """
        mae = self.mae_loss(judge_scores, human_scores)
        
        if loss_type == 'pairwise':
            rank_loss = self.pairwise_ranking_loss(judge_scores, human_scores)
        elif loss_type == 'kendall':
            rank_loss = self.kendall_tau_loss(judge_scores, human_scores)
        elif loss_type == 'spearman':
            rank_loss = self.spearman_loss(judge_scores, human_scores)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        total_loss = self.alpha * mae + self.beta * rank_loss
        
        return total_loss, mae, rank_loss
