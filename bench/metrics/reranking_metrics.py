"""Reranking metrics calculation for RAG system evaluation.

This module implements metrics for evaluating reranking effectiveness including
ranking improvement, MAP, position improvement, and reranking effectiveness.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Union


@dataclass
class RerankingMetrics:
    """Container for reranking evaluation metrics."""
    
    ranking_improvement: float
    map_score: float
    position_improvement: float
    reranking_effectiveness: float
    ndcg_improvement: Dict[int, float]
    precision_improvement: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Union[float, Dict[int, float]]]:
        """Convert metrics to dictionary format."""
        return {
            "ranking_improvement": self.ranking_improvement,
            "map_score": self.map_score,
            "position_improvement": self.position_improvement,
            "reranking_effectiveness": self.reranking_effectiveness,
            "ndcg_improvement": self.ndcg_improvement,
            "precision_improvement": self.precision_improvement,
        }


def calculate_ranking_improvement(
    original_ranking: List[str],
    reranked_ranking: List[str],
    relevant_docs: Set[str],
    k: int = 10,
) -> float:
    """Calculate improvement in ranking quality after reranking.
    
    Measures how much the ranking of relevant documents improved.
    
    Args:
        original_ranking: Original document ranking before reranking
        reranked_ranking: Document ranking after reranking
        relevant_docs: Set of relevant document IDs
        k: Cut-off rank for evaluation
        
    Returns:
        Ranking improvement score (positive = improvement, negative = degradation)
    """
    if k <= 0:
        return 0.0
    
    # Calculate average position of relevant docs in both rankings
    original_positions = []
    reranked_positions = []
    
    for doc_id in relevant_docs:
        # Find position in original ranking
        try:
            orig_pos = original_ranking.index(doc_id) + 1  # 1-indexed
            if orig_pos <= k:
                original_positions.append(orig_pos)
        except ValueError:
            # Document not in original ranking
            pass
            
        # Find position in reranked ranking
        try:
            rerank_pos = reranked_ranking.index(doc_id) + 1  # 1-indexed
            if rerank_pos <= k:
                reranked_positions.append(rerank_pos)
        except ValueError:
            # Document not in reranked ranking
            pass
    
    if not original_positions and not reranked_positions:
        return 0.0
    
    # Calculate average positions (lower is better)
    avg_original = sum(original_positions) / len(original_positions) if original_positions else k + 1
    avg_reranked = sum(reranked_positions) / len(reranked_positions) if reranked_positions else k + 1
    
    # Improvement = reduction in average position
    # Normalize by k to get a score between -1 and 1
    improvement = (avg_original - avg_reranked) / k
    
    return improvement


def calculate_map(
    retrieved_docs_list: List[List[str]],
    relevant_docs_list: List[Set[str]],
) -> float:
    """Calculate Mean Average Precision (MAP).
    
    MAP is the mean of Average Precision scores across all queries.
    
    Args:
        retrieved_docs_list: List of retrieved document lists for each query
        relevant_docs_list: List of relevant document sets for each query
        
    Returns:
        Mean Average Precision score
    """
    if len(retrieved_docs_list) != len(relevant_docs_list):
        raise ValueError("Number of queries must match")
    
    if len(retrieved_docs_list) == 0:
        return 0.0
    
    average_precisions = []
    
    for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
        ap = calculate_average_precision(retrieved_docs, relevant_docs)
        average_precisions.append(ap)
    
    return sum(average_precisions) / len(average_precisions)


def calculate_average_precision(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
) -> float:
    """Calculate Average Precision for a single query.
    
    AP = (sum of precisions at relevant document positions) / (number of relevant docs)
    
    Args:
        retrieved_docs: List of retrieved document IDs in ranked order
        relevant_docs: Set of relevant document IDs
        
    Returns:
        Average Precision score for the query
    """
    if len(relevant_docs) == 0:
        return 0.0
    
    precisions_at_relevant = []
    relevant_count = 0
    
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)  # i+1 for 1-indexed position
            precisions_at_relevant.append(precision_at_i)
    
    if not precisions_at_relevant:
        return 0.0
    
    return sum(precisions_at_relevant) / len(relevant_docs)


def calculate_position_improvement(
    original_rankings: List[List[str]],
    reranked_rankings: List[List[str]],
    relevant_docs_list: List[Set[str]],
) -> float:
    """Calculate average position improvement for relevant documents.
    
    Args:
        original_rankings: Original document rankings before reranking
        reranked_rankings: Document rankings after reranking
        relevant_docs_list: List of relevant document sets for each query
        
    Returns:
        Average position improvement (positive = improvement)
    """
    if len(original_rankings) != len(reranked_rankings) != len(relevant_docs_list):
        raise ValueError("All input lists must have the same length")
    
    if len(original_rankings) == 0:
        return 0.0
    
    position_improvements = []
    
    for orig_ranking, rerank_ranking, relevant_docs in zip(
        original_rankings, reranked_rankings, relevant_docs_list
    ):
        for doc_id in relevant_docs:
            # Find positions in both rankings
            orig_pos = None
            rerank_pos = None
            
            try:
                orig_pos = orig_ranking.index(doc_id) + 1  # 1-indexed
            except ValueError:
                pass
                
            try:
                rerank_pos = rerank_ranking.index(doc_id) + 1  # 1-indexed
            except ValueError:
                pass
            
            # Calculate improvement if document appears in both rankings
            if orig_pos is not None and rerank_pos is not None:
                improvement = orig_pos - rerank_pos  # Positive = moved up
                position_improvements.append(improvement)
    
    if not position_improvements:
        return 0.0
    
    return sum(position_improvements) / len(position_improvements)


def calculate_reranking_effectiveness(
    original_metrics: Dict[str, float],
    reranked_metrics: Dict[str, float],
    weights: Dict[str, float] = None,
) -> float:
    """Calculate overall reranking effectiveness.
    
    Combines multiple metrics improvements into a single effectiveness score.
    
    Args:
        original_metrics: Metrics before reranking
        reranked_metrics: Metrics after reranking
        weights: Optional weights for different metrics
        
    Returns:
        Overall reranking effectiveness score
    """
    if weights is None:
        weights = {
            "ndcg_at_10": 0.3,
            "precision_at_10": 0.2,
            "recall_at_10": 0.2,
            "mrr": 0.3,
        }
    
    effectiveness = 0.0
    total_weight = 0.0
    
    for metric_name, weight in weights.items():
        if metric_name in original_metrics and metric_name in reranked_metrics:
            original_value = original_metrics[metric_name]
            reranked_value = reranked_metrics[metric_name]
            
            # Calculate relative improvement
            if original_value > 0:
                improvement = (reranked_value - original_value) / original_value
            else:
                improvement = 1.0 if reranked_value > 0 else 0.0
            
            effectiveness += weight * improvement
            total_weight += weight
    
    return effectiveness / total_weight if total_weight > 0 else 0.0


def calculate_ndcg_improvement(
    original_rankings: List[List[str]],
    reranked_rankings: List[List[str]],
    relevant_docs_list: List[Set[str]],
    k_values: List[int] = None,
    relevance_scores_list: List[Dict[str, float]] = None,
) -> Dict[int, float]:
    """Calculate NDCG improvement at different k values.
    
    Args:
        original_rankings: Original document rankings before reranking
        reranked_rankings: Document rankings after reranking
        relevant_docs_list: List of relevant document sets for each query
        k_values: List of k values to calculate improvement for
        relevance_scores_list: Optional list of relevance score dicts
        
    Returns:
        Dict mapping k values to NDCG improvement scores
    """
    if k_values is None:
        k_values = [5, 10, 20]
    
    from .retrieval_metrics import calculate_ndcg_at_k
    
    improvements = {}
    
    for k in k_values:
        original_ndcgs = []
        reranked_ndcgs = []
        
        for i, (orig_ranking, rerank_ranking, relevant_docs) in enumerate(
            zip(original_rankings, reranked_rankings, relevant_docs_list)
        ):
            relevance_scores = relevance_scores_list[i] if relevance_scores_list else None
            
            orig_ndcg = calculate_ndcg_at_k(orig_ranking, relevant_docs, k, relevance_scores)
            rerank_ndcg = calculate_ndcg_at_k(rerank_ranking, relevant_docs, k, relevance_scores)
            
            original_ndcgs.append(orig_ndcg)
            reranked_ndcgs.append(rerank_ndcg)
        
        # Calculate average improvement
        if original_ndcgs and reranked_ndcgs:
            avg_original = sum(original_ndcgs) / len(original_ndcgs)
            avg_reranked = sum(reranked_ndcgs) / len(reranked_ndcgs)
            
            if avg_original > 0:
                improvement = (avg_reranked - avg_original) / avg_original
            else:
                improvement = 1.0 if avg_reranked > 0 else 0.0
                
            improvements[k] = improvement
        else:
            improvements[k] = 0.0
    
    return improvements


def calculate_precision_improvement(
    original_rankings: List[List[str]],
    reranked_rankings: List[List[str]],
    relevant_docs_list: List[Set[str]],
    k_values: List[int] = None,
) -> Dict[int, float]:
    """Calculate precision improvement at different k values.
    
    Args:
        original_rankings: Original document rankings before reranking
        reranked_rankings: Document rankings after reranking
        relevant_docs_list: List of relevant document sets for each query
        k_values: List of k values to calculate improvement for
        
    Returns:
        Dict mapping k values to precision improvement scores
    """
    if k_values is None:
        k_values = [5, 10, 20]
    
    from .retrieval_metrics import calculate_precision_at_k
    
    improvements = {}
    
    for k in k_values:
        original_precisions = []
        reranked_precisions = []
        
        for orig_ranking, rerank_ranking, relevant_docs in zip(
            original_rankings, reranked_rankings, relevant_docs_list
        ):
            orig_precision = calculate_precision_at_k(orig_ranking, relevant_docs, k)
            rerank_precision = calculate_precision_at_k(rerank_ranking, relevant_docs, k)
            
            original_precisions.append(orig_precision)
            reranked_precisions.append(rerank_precision)
        
        # Calculate average improvement
        if original_precisions and reranked_precisions:
            avg_original = sum(original_precisions) / len(original_precisions)
            avg_reranked = sum(reranked_precisions) / len(reranked_precisions)
            
            if avg_original > 0:
                improvement = (avg_reranked - avg_original) / avg_original
            else:
                improvement = 1.0 if avg_reranked > 0 else 0.0
                
            improvements[k] = improvement
        else:
            improvements[k] = 0.0
    
    return improvements


def calculate_all_reranking_metrics(
    original_rankings: List[List[str]],
    reranked_rankings: List[List[str]],
    relevant_docs_list: List[Set[str]],
    k_values: List[int] = None,
    relevance_scores_list: List[Dict[str, float]] = None,
) -> RerankingMetrics:
    """Calculate all reranking metrics for a set of queries.
    
    Args:
        original_rankings: Original document rankings before reranking
        reranked_rankings: Document rankings after reranking
        relevant_docs_list: List of relevant document sets for each query
        k_values: List of k values to calculate metrics for
        relevance_scores_list: Optional list of relevance score dicts
        
    Returns:
        RerankingMetrics object with all calculated metrics
    """
    if k_values is None:
        k_values = [5, 10, 20]
    
    if len(original_rankings) != len(reranked_rankings) != len(relevant_docs_list):
        raise ValueError("All input lists must have the same length")
    
    if len(original_rankings) == 0:
        return RerankingMetrics(
            ranking_improvement=0.0,
            map_score=0.0,
            position_improvement=0.0,
            reranking_effectiveness=0.0,
            ndcg_improvement={k: 0.0 for k in k_values},
            precision_improvement={k: 0.0 for k in k_values},
        )
    
    # Calculate individual metrics
    ranking_improvement = sum(
        calculate_ranking_improvement(orig, rerank, relevant, k=10)
        for orig, rerank, relevant in zip(original_rankings, reranked_rankings, relevant_docs_list)
    ) / len(original_rankings)
    
    map_score = calculate_map(reranked_rankings, relevant_docs_list)
    position_improvement = calculate_position_improvement(
        original_rankings, reranked_rankings, relevant_docs_list
    )
    
    ndcg_improvement = calculate_ndcg_improvement(
        original_rankings, reranked_rankings, relevant_docs_list, k_values, relevance_scores_list
    )
    
    precision_improvement = calculate_precision_improvement(
        original_rankings, reranked_rankings, relevant_docs_list, k_values
    )
    
    # Calculate overall effectiveness (placeholder - would need to compute original metrics)
    reranking_effectiveness = 0.0  # Would be calculated with original vs reranked metric comparison
    
    return RerankingMetrics(
        ranking_improvement=ranking_improvement,
        map_score=map_score,
        position_improvement=position_improvement,
        reranking_effectiveness=reranking_effectiveness,
        ndcg_improvement=ndcg_improvement,
        precision_improvement=precision_improvement,
    )