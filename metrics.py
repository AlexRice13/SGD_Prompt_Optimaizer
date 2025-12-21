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
