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


if __name__ == '__main__':
    """Unit tests for LossFunctions class."""
    import numpy as np
    
    print("Running LossFunctions unit tests...")
    
    # Test data
    judge_scores = np.array([8.0, 6.5, 7.0, 9.0, 5.5])
    human_scores = np.array([7.5, 6.0, 7.5, 8.5, 6.0])
    
    # Test 1: MAE loss
    print("\n1. Testing MAE loss...")
    mae = LossFunctions.mae_loss(judge_scores, human_scores)
    expected = np.mean(np.abs(judge_scores - human_scores))
    assert abs(mae - expected) < 1e-6
    print(f"   MAE: {mae:.4f} ✓")
    
    # Test 2: Huber loss
    print("\n2. Testing Huber loss...")
    huber = LossFunctions.huber_loss(judge_scores, human_scores, delta=1.0)
    assert huber >= 0
    print(f"   Huber loss: {huber:.4f} ✓")
    
    # Test 3: Pairwise ranking loss
    print("\n3. Testing pairwise ranking loss...")
    rank_loss = LossFunctions.pairwise_ranking_loss(judge_scores, human_scores)
    assert rank_loss >= 0
    print(f"   Pairwise ranking loss: {rank_loss:.4f} ✓")
    
    # Test 4: Kendall tau loss
    print("\n4. Testing Kendall tau loss...")
    tau_loss = LossFunctions.kendall_tau_loss(judge_scores, human_scores)
    assert 0 <= tau_loss <= 2.0
    print(f"   Kendall tau loss: {tau_loss:.4f} ✓")
    
    # Test 5: Spearman loss
    print("\n5. Testing Spearman loss...")
    spearman_loss = LossFunctions.spearman_loss(judge_scores, human_scores)
    assert 0 <= spearman_loss <= 2.0
    print(f"   Spearman loss: {spearman_loss:.4f} ✓")
    
    # Test 6: Combined loss
    print("\n6. Testing combined loss...")
    loss_fn = LossFunctions(alpha=1.0, beta=1.0)
    total, mae_comp, rank_comp = loss_fn.compute_loss(judge_scores, human_scores, 'pairwise')
    assert total >= 0
    assert mae_comp >= 0
    assert rank_comp >= 0
    assert abs(total - (mae_comp + rank_comp)) < 1e-6
    print(f"   Total: {total:.4f}, MAE: {mae_comp:.4f}, Rank: {rank_comp:.4f} ✓")
    
    # Test 7: Different loss types
    print("\n7. Testing different loss types...")
    for loss_type in ['pairwise', 'kendall', 'spearman']:
        total, _, _ = loss_fn.compute_loss(judge_scores, human_scores, loss_type)
        assert total >= 0
        print(f"   {loss_type}: {total:.4f} ✓")
    
    # Test 8: Edge case - single sample
    print("\n8. Testing edge cases...")
    single_judge = np.array([5.0])
    single_human = np.array([5.5])
    mae_single = LossFunctions.mae_loss(single_judge, single_human)
    assert mae_single == 0.5
    rank_single = LossFunctions.pairwise_ranking_loss(single_judge, single_human)
    assert rank_single == 0.0
    print("   Single sample handled correctly ✓")
    
    # Test 9: Custom weights
    print("\n9. Testing custom weights...")
    loss_fn2 = LossFunctions(alpha=2.0, beta=0.5)
    total, mae_comp, rank_comp = loss_fn2.compute_loss(judge_scores, human_scores, 'pairwise')
    expected_total = 2.0 * mae_comp + 0.5 * rank_comp
    assert abs(total - expected_total) < 1e-6
    print(f"   Custom weights work correctly ✓")
    
    print("\n" + "="*50)
    print("All LossFunctions tests passed! ✓")
    print("="*50)
