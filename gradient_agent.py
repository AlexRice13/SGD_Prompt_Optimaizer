"""
Gradient Agent: Construct proxy gradients in language space.

Implements gradient construction with information isolation constraints,
using only aggregated statistics without access to specific sample content.
"""

from typing import List, Dict, Tuple, Callable
import numpy as np


class GradientAgent:
    """
    Constructs proxy gradients using aggregated statistics.
    
    Implements information isolation: does not access actual response content,
    only receives aggregated error statistics.
    """
    
    def __init__(self, llm_fn: Callable[[str], str], 
                 n_samples_per_category: int = 3):
        """
        Initialize gradient agent.
        
        Args:
            llm_fn: Function that takes a prompt and returns text
            n_samples_per_category: Number of samples to select from each category
        """
        self.llm_fn = llm_fn
        self.n_samples_per_category = n_samples_per_category
    
    def select_gradient_samples(self, judge_scores: np.ndarray, 
                               human_scores: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Select samples for gradient construction.
        
        Selects three categories:
        1. Overestimated: JudgeScore >> HumanScore
        2. Underestimated: JudgeScore << HumanScore
        3. Well-aligned: JudgeScore ≈ HumanScore
        
        Args:
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            
        Returns:
            Dictionary with indices for each category
        """
        errors = judge_scores - human_scores
        abs_errors = np.abs(errors)
        
        # Category 1: Overestimated (positive error)
        overestimated_idx = np.where(errors > 0)[0]
        if len(overestimated_idx) > 0:
            # Sort by error magnitude
            sorted_idx = overestimated_idx[np.argsort(-errors[overestimated_idx])]
            overestimated = sorted_idx[:self.n_samples_per_category]
        else:
            overestimated = np.array([])
        
        # Category 2: Underestimated (negative error)
        underestimated_idx = np.where(errors < 0)[0]
        if len(underestimated_idx) > 0:
            # Sort by error magnitude
            sorted_idx = underestimated_idx[np.argsort(errors[underestimated_idx])]
            underestimated = sorted_idx[:self.n_samples_per_category]
        else:
            underestimated = np.array([])
        
        # Category 3: Well-aligned (small error)
        sorted_by_abs_error = np.argsort(abs_errors)
        well_aligned = sorted_by_abs_error[:self.n_samples_per_category]
        
        return {
            'overestimated': overestimated,
            'underestimated': underestimated,
            'well_aligned': well_aligned
        }
    
    def compute_statistics(self, judge_scores: np.ndarray, 
                          human_scores: np.ndarray,
                          selected_indices: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Compute aggregated statistics (information isolation constraint).
        
        Returns only aggregated statistics, not individual sample details.
        
        Args:
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            selected_indices: Indices for each category
            
        Returns:
            Aggregated statistics dictionary
        """
        errors = judge_scores - human_scores
        
        stats = {
            'total_samples': len(judge_scores),
            'overestimated': {
                'count': len(selected_indices['overestimated']),
                'mean_error': float(np.mean(errors[selected_indices['overestimated']])) 
                             if len(selected_indices['overestimated']) > 0 else 0.0,
                'max_error': float(np.max(errors[selected_indices['overestimated']]))
                            if len(selected_indices['overestimated']) > 0 else 0.0,
            },
            'underestimated': {
                'count': len(selected_indices['underestimated']),
                'mean_error': float(np.mean(errors[selected_indices['underestimated']]))
                             if len(selected_indices['underestimated']) > 0 else 0.0,
                'min_error': float(np.min(errors[selected_indices['underestimated']]))
                            if len(selected_indices['underestimated']) > 0 else 0.0,
            },
            'well_aligned': {
                'count': len(selected_indices['well_aligned']),
                'mean_abs_error': float(np.mean(np.abs(errors[selected_indices['well_aligned']])))
                                 if len(selected_indices['well_aligned']) > 0 else 0.0,
            },
            'overall': {
                'mae': float(np.mean(np.abs(errors))),
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
            }
        }
        
        return stats
    
    def construct_proxy_gradient_with_samples(self, current_prompt: str,
                                              statistics: Dict[str, any],
                                              selected_indices: Dict[str, np.ndarray],
                                              judge_scores: np.ndarray,
                                              human_scores: np.ndarray,
                                              responses: List[str]) -> str:
        """
        Construct proxy gradient using LLM with sample content reference.
        
        Provides actual sample content for better context, but requires
        abstract optimization suggestions without including specific content.
        
        Args:
            current_prompt: Current JudgePrompt text
            statistics: Aggregated statistics from compute_statistics
            selected_indices: Selected sample indices by category
            judge_scores: All judge scores
            human_scores: All human scores
            responses: All response texts
            
        Returns:
            Proxy gradient as structured text
        """
        # Build sample content strings
        def format_samples(indices, category_name):
            if len(indices) == 0:
                return f"  （无{category_name}样本）\n"
            
            sample_text = ""
            for i, idx in enumerate(indices[:self.n_samples_per_category], 1):
                response = responses[idx] if idx < len(responses) else "[缺失]"
                # Truncate long responses for readability
                if len(response) > 200:
                    response = response[:200] + "..."
                error = judge_scores[idx] - human_scores[idx]
                sample_text += f"  样本{i}: {response}\n"
                sample_text += f"    人工评分: {human_scores[idx]:.1f}, AI评分: {judge_scores[idx]:.1f}, 误差: {error:+.1f}\n"
            return sample_text
        
        overestimated_samples = format_samples(selected_indices['overestimated'], "高估")
        underestimated_samples = format_samples(selected_indices['underestimated'], "低估")
        well_aligned_samples = format_samples(selected_indices['well_aligned'], "对齐")
        
        gradient_prompt = f"""你是一个元优化器，正在分析评分prompt的表现。

当前评分Prompt：
{current_prompt}

性能统计：
- 总评估样本数: {statistics['total_samples']}
- 整体MAE: {statistics['overall']['mae']:.3f}
- 平均误差（偏置）: {statistics['overall']['mean_error']:.3f}
- 误差标准差: {statistics['overall']['std_error']:.3f}

误差分类分析：
1. 高估样本（AI评分过高）：
   - 数量: {statistics['overestimated']['count']}
   - 平均误差: +{statistics['overestimated']['mean_error']:.3f}
   - 最大误差: +{statistics['overestimated']['max_error']:.3f}

2. 低估样本（AI评分过低）：
   - 数量: {statistics['underestimated']['count']}
   - 平均误差: {statistics['underestimated']['mean_error']:.3f}
   - 最小误差: {statistics['underestimated']['min_error']:.3f}

3. 对齐样本（AI评分接近人工）：
   - 数量: {statistics['well_aligned']['count']}
   - 平均绝对误差: {statistics['well_aligned']['mean_abs_error']:.3f}

参考样本（用于理解误差模式，但不要在建议中引用具体内容）：

高估样本（评分过高）：
{overestimated_samples}
低估样本（评分过低）：
{underestimated_samples}
对齐样本（评分准确）：
{well_aligned_samples}

任务：基于统计数据和上述样本，提供抽象的优化建议：
1. 识别prompt中可能导致高估的抽象模式
2. 识别prompt中可能导致低估的抽象模式
3. 提出prompt改进的方向性建议

输出结构化分析：
- 高估原因分析: 为什么某些类型的回答可能被评分过高（抽象分析，不引用具体样本）
- 低估原因分析: 为什么某些类型的回答可能被评分过低（抽象分析，不引用具体样本）
- 改进方向: prompt修改的高层次指导方向

重要：你的建议必须是抽象和可泛化的，不要包含上述具体样本的内容。"""

        return self.llm_fn(gradient_prompt)
    
    def compute_gradient(self, current_prompt: str,
                        judge_scores: np.ndarray,
                        human_scores: np.ndarray,
                        responses: List[str]) -> Dict[str, any]:
        """
        Full gradient computation pipeline.
        
        Args:
            current_prompt: Current JudgePrompt text
            judge_scores: Scores from Judge LLM
            human_scores: Human reference scores
            responses: List of response texts for reference
            
        Returns:
            Dictionary containing statistics and proxy gradient
        """
        # Select samples
        selected_indices = self.select_gradient_samples(judge_scores, human_scores)
        
        # Compute aggregated statistics
        statistics = self.compute_statistics(judge_scores, human_scores, selected_indices)
        
        # Construct proxy gradient with sample content
        proxy_gradient = self.construct_proxy_gradient_with_samples(
            current_prompt, statistics, selected_indices, 
            judge_scores, human_scores, responses
        )
        
        return {
            'statistics': statistics,
            'selected_indices': selected_indices,
            'proxy_gradient': proxy_gradient
        }
