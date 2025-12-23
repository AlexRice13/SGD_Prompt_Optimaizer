"""
Metrics: Evaluation metrics for monitoring optimization.

Implements various metrics for tracking prompt optimization progress
and detecting degeneration.
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr, entropy


class Metrics:
    """
    Metrics for monitoring optimization progress.
    """
    
    @staticmethod
    def mae(judge_scores: np.ndarray, human_scores: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(judge_scores - human_scores)))
    
    @staticmethod
    def mse(judge_scores: np.ndarray, human_scores: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((judge_scores - human_scores) ** 2))
    
    @staticmethod
    def rmse(judge_scores: np.ndarray, human_scores: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((judge_scores - human_scores) ** 2)))
    
    @staticmethod
    def kendall_tau(judge_scores: np.ndarray, human_scores: np.ndarray) -> float:
        """Kendall's Tau rank correlation."""
        if len(judge_scores) < 2:
            return 0.0
        tau, _ = kendalltau(judge_scores, human_scores)
        return float(tau) if not np.isnan(tau) else 0.0
    
    @staticmethod
    def spearman_rho(judge_scores: np.ndarray, human_scores: np.ndarray) -> float:
        """Spearman's rank correlation."""
        if len(judge_scores) < 2:
            return 0.0
        rho, _ = spearmanr(judge_scores, human_scores)
        return float(rho) if not np.isnan(rho) else 0.0
    
    @staticmethod
    def score_entropy(judge_scores: np.ndarray, n_bins: int = 10) -> float:
        """
        Entropy of score distribution.
        
        Low entropy indicates score collapse (all scores similar).
        
        Args:
            judge_scores: Scores from Judge LLM
            n_bins: Number of bins for histogram
            
        Returns:
            Entropy value
        """
        if len(judge_scores) < 2:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(judge_scores, bins=n_bins)
        # Normalize to probability distribution
        hist = hist / len(judge_scores)
        # Filter out zero bins
        hist = hist[hist > 0]
        
        return float(entropy(hist))
    
    @staticmethod
    def score_std(judge_scores: np.ndarray) -> float:
        """Standard deviation of scores."""
        return float(np.std(judge_scores))
    
    @staticmethod
    def mean_self_consistency_variance(variances: np.ndarray) -> float:
        """
        Mean self-consistency variance across samples.
        
        High variance indicates Judge is inconsistent.
        
        Args:
            variances: Self-consistency variances from forward pass
            
        Returns:
            Mean variance
        """
        return float(np.mean(variances))
    
    @staticmethod
    def compute_all_metrics(judge_scores: np.ndarray, 
                          human_scores: np.ndarray,
                          variances: np.ndarray = None) -> dict:
        """
        Compute all metrics.
        
        Args:
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            variances: Optional self-consistency variances
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'mae': Metrics.mae(judge_scores, human_scores),
            'mse': Metrics.mse(judge_scores, human_scores),
            'rmse': Metrics.rmse(judge_scores, human_scores),
            'kendall_tau': Metrics.kendall_tau(judge_scores, human_scores),
            'spearman_rho': Metrics.spearman_rho(judge_scores, human_scores),
            'score_entropy': Metrics.score_entropy(judge_scores),
            'score_std': Metrics.score_std(judge_scores),
        }
        
        if variances is not None:
            metrics['mean_self_consistency_var'] = Metrics.mean_self_consistency_variance(variances)
        
        return metrics


if __name__ == '__main__':
    """Unit tests for Metrics class."""
    import numpy as np
    
    print("Running Metrics unit tests...")
    
    # Test data
    judge_scores = np.array([8.0, 6.5, 7.0, 9.0, 5.5])
    human_scores = np.array([7.5, 6.0, 7.5, 8.5, 6.0])
    variances = np.array([0.1, 0.2, 0.15, 0.05, 0.3])
    
    # Test 1: MAE
    print("\n1. Testing MAE...")
    mae = Metrics.mae(judge_scores, human_scores)
    expected_mae = np.mean(np.abs(judge_scores - human_scores))
    assert abs(mae - expected_mae) < 1e-6
    print(f"   MAE: {mae:.4f} ✓")
    
    # Test 2: MSE and RMSE
    print("\n2. Testing MSE and RMSE...")
    mse = Metrics.mse(judge_scores, human_scores)
    rmse = Metrics.rmse(judge_scores, human_scores)
    assert mse >= 0
    assert rmse == np.sqrt(mse)
    print(f"   MSE: {mse:.4f}, RMSE: {rmse:.4f} ✓")
    
    # Test 3: Kendall Tau
    print("\n3. Testing Kendall Tau...")
    tau = Metrics.kendall_tau(judge_scores, human_scores)
    assert -1.0 <= tau <= 1.0
    print(f"   Kendall Tau: {tau:.4f} ✓")
    
    # Test 4: Spearman Rho
    print("\n4. Testing Spearman Rho...")
    rho = Metrics.spearman_rho(judge_scores, human_scores)
    assert -1.0 <= rho <= 1.0
    print(f"   Spearman Rho: {rho:.4f} ✓")
    
    # Test 5: Score entropy
    print("\n5. Testing score entropy...")
    entropy = Metrics.score_entropy(judge_scores)
    assert entropy >= 0
    print(f"   Entropy: {entropy:.4f} ✓")
    
    # Test 6: Score std
    print("\n6. Testing score std...")
    std = Metrics.score_std(judge_scores)
    assert std >= 0
    assert abs(std - np.std(judge_scores)) < 1e-6
    print(f"   Std: {std:.4f} ✓")
    
    # Test 7: Self-consistency variance
    print("\n7. Testing self-consistency variance...")
    mean_var = Metrics.mean_self_consistency_variance(variances)
    assert mean_var >= 0
    assert abs(mean_var - np.mean(variances)) < 1e-6
    print(f"   Mean variance: {mean_var:.4f} ✓")
    
    # Test 8: Compute all metrics
    print("\n8. Testing compute_all_metrics...")
    all_metrics = Metrics.compute_all_metrics(judge_scores, human_scores, variances)
    assert 'mae' in all_metrics
    assert 'kendall_tau' in all_metrics
    assert 'mean_self_consistency_var' in all_metrics
    print(f"   All metrics computed: {len(all_metrics)} metrics ✓")
    
    # Test 9: Edge case - single sample
    print("\n9. Testing edge cases...")
    single = np.array([5.0])
    tau_single = Metrics.kendall_tau(single, single)
    assert tau_single == 0.0
    print("   Single sample handled correctly ✓")
    
    print("\n" + "="*50)
    print("All Metrics tests passed! ✓")
    print("="*50)
