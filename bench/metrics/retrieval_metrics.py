"""Retrieval metrics calculation for RAG system evaluation.

This module implements standard Information Retrieval metrics including
precision@k, recall@k, NDCG@k, MRR, hit rate, and F1@k.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Set, Union, Optional


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float
    hit_rate: float
    f1_at_k: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Union[float, Dict[int, float]]]:
        """Convert metrics to dictionary format."""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": self.mrr,
            "hit_rate": self.hit_rate,
            "f1_at_k": self.f1_at_k,
        }


def calculate_precision_at_k(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int,
) -> float:
    """Calculate precision at k.
    
    Precision@k = (# of relevant documents in top-k) / k
    
    Args:
        retrieved_docs: List of retrieved document IDs in ranked order
        relevant_docs: Set of relevant document IDs
        k: Cut-off rank
        
    Returns:
        Precision at k value (0.0 to 1.0)
    """
    if k <= 0:
        return 0.0
        
    # Take only top-k retrieved documents
    top_k_docs = retrieved_docs[:k]
    
    # Count relevant documents in top-k
    relevant_in_top_k = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)
    
    return relevant_in_top_k / k


def calculate_recall_at_k(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int,
) -> float:
    """Calculate recall at k.
    
    Recall@k = (# of relevant documents in top-k) / (total # of relevant documents)
    
    Args:
        retrieved_docs: List of retrieved document IDs in ranked order
        relevant_docs: Set of relevant document IDs
        k: Cut-off rank
        
    Returns:
        Recall at k value (0.0 to 1.0)
    """
    if k <= 0 or len(relevant_docs) == 0:
        return 0.0
        
    # Take only top-k retrieved documents
    top_k_docs = retrieved_docs[:k]
    
    # Count relevant documents in top-k
    relevant_in_top_k = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)
    
    return relevant_in_top_k / len(relevant_docs)


def calculate_ndcg_at_k(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int,
    relevance_scores: Optional[Dict[str, float]] = None,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.
    
    NDCG@k considers both relevance and ranking position with logarithmic
    discount for lower positions.
    
    Args:
        retrieved_docs: List of retrieved document IDs in ranked order
        relevant_docs: Set of relevant document IDs
        k: Cut-off rank
        relevance_scores: Optional dict mapping doc_id to relevance score.
                         If not provided, binary relevance (0/1) is used.
        
    Returns:
        NDCG at k value (0.0 to 1.0)
    """
    if k <= 0 or len(relevant_docs) == 0:
        return 0.0
        
    # Take only top-k retrieved documents
    top_k_docs = retrieved_docs[:k]
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, doc_id in enumerate(top_k_docs):
        rank = i + 1  # 1-indexed rank
        
        if doc_id in relevant_docs:
            # Get relevance score (default to 1.0 for binary relevance)
            relevance = relevance_scores.get(doc_id, 1.0) if relevance_scores else 1.0
            
            # DCG formula: sum(rel_i / log2(i + 1))
            dcg += relevance / math.log2(rank + 1)
    
    # Calculate IDCG (Ideal DCG) - best possible ranking
    # Sort relevant docs by relevance score (descending)
    if relevance_scores:
        relevant_with_scores = [
            (doc_id, relevance_scores.get(doc_id, 1.0))
            for doc_id in relevant_docs
        ]
        relevant_with_scores.sort(key=lambda x: x[1], reverse=True)
        ideal_docs = [doc_id for doc_id, _ in relevant_with_scores[:k]]
    else:
        # For binary relevance, any order of relevant docs is ideal
        ideal_docs = list(relevant_docs)[:k]
    
    idcg = 0.0
    for i, doc_id in enumerate(ideal_docs):
        rank = i + 1
        relevance = relevance_scores.get(doc_id, 1.0) if relevance_scores else 1.0
        idcg += relevance / math.log2(rank + 1)
    
    # NDCG = DCG / IDCG
    return dcg / idcg if idcg > 0 else 0.0


def calculate_mrr(
    retrieved_docs_list: List[List[str]],
    relevant_docs_list: List[Set[str]],
) -> float:
    """Calculate Mean Reciprocal Rank across multiple queries.
    
    MRR = (1/|Q|) * sum(1/rank of first relevant document)
    
    Args:
        retrieved_docs_list: List of retrieved document lists for each query
        relevant_docs_list: List of relevant document sets for each query
        
    Returns:
        Mean Reciprocal Rank value
    """
    if len(retrieved_docs_list) != len(relevant_docs_list):
        raise ValueError("Number of queries must match between retrieved and relevant docs")
        
    if len(retrieved_docs_list) == 0:
        return 0.0
    
    reciprocal_ranks = []
    
    for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
        # Find rank of first relevant document
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # No relevant document found
            reciprocal_ranks.append(0.0)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_hit_rate(
    retrieved_docs_list: List[List[str]],
    relevant_docs_list: List[Set[str]],
    k: int = None,
) -> float:
    """Calculate hit rate (percentage of queries with at least one relevant result).
    
    Hit Rate = (# of queries with at least one relevant result) / (total # of queries)
    
    Args:
        retrieved_docs_list: List of retrieved document lists for each query
        relevant_docs_list: List of relevant document sets for each query
        k: Optional cut-off rank. If None, considers all retrieved docs.
        
    Returns:
        Hit rate value (0.0 to 1.0)
    """
    if len(retrieved_docs_list) != len(relevant_docs_list):
        raise ValueError("Number of queries must match between retrieved and relevant docs")
        
    if len(retrieved_docs_list) == 0:
        return 0.0
    
    hits = 0
    
    for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
        # Consider only top-k if specified
        docs_to_check = retrieved_docs[:k] if k is not None else retrieved_docs
        
        # Check if any retrieved document is relevant
        if any(doc_id in relevant_docs for doc_id in docs_to_check):
            hits += 1
    
    return hits / len(retrieved_docs_list)


def calculate_f1_at_k(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int,
) -> float:
    """Calculate F1 score at k.
    
    F1@k = 2 * (Precision@k * Recall@k) / (Precision@k + Recall@k)
    
    Args:
        retrieved_docs: List of retrieved document IDs in ranked order
        relevant_docs: Set of relevant document IDs
        k: Cut-off rank
        
    Returns:
        F1 at k value (0.0 to 1.0)
    """
    precision = calculate_precision_at_k(retrieved_docs, relevant_docs, k)
    recall = calculate_recall_at_k(retrieved_docs, relevant_docs, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def calculate_all_retrieval_metrics(
    retrieved_docs_list: List[List[str]],
    relevant_docs_list: List[Set[str]],
    k_values: List[int] = None,
    relevance_scores_list: List[Dict[str, float]] = None,
) -> RetrievalMetrics:
    """Calculate all retrieval metrics for a set of queries.
    
    Args:
        retrieved_docs_list: List of retrieved document lists for each query
        relevant_docs_list: List of relevant document sets for each query
        k_values: List of k values to calculate metrics for
        relevance_scores_list: Optional list of relevance score dicts
        
    Returns:
        RetrievalMetrics object with all calculated metrics
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]
    
    if len(retrieved_docs_list) != len(relevant_docs_list):
        raise ValueError("Number of queries must match")
    
    num_queries = len(retrieved_docs_list)
    if num_queries == 0:
        return RetrievalMetrics(
            precision_at_k={k: 0.0 for k in k_values},
            recall_at_k={k: 0.0 for k in k_values},
            ndcg_at_k={k: 0.0 for k in k_values},
            mrr=0.0,
            hit_rate=0.0,
            f1_at_k={k: 0.0 for k in k_values},
        )
    
    # Calculate metrics for each k
    precision_at_k = {}
    recall_at_k = {}
    ndcg_at_k = {}
    f1_at_k = {}
    
    for k in k_values:
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        f1_scores = []
        
        for i, (retrieved_docs, relevant_docs) in enumerate(zip(retrieved_docs_list, relevant_docs_list)):
            relevance_scores = relevance_scores_list[i] if relevance_scores_list else None
            
            precision_scores.append(calculate_precision_at_k(retrieved_docs, relevant_docs, k))
            recall_scores.append(calculate_recall_at_k(retrieved_docs, relevant_docs, k))
            ndcg_scores.append(calculate_ndcg_at_k(retrieved_docs, relevant_docs, k, relevance_scores))
            f1_scores.append(calculate_f1_at_k(retrieved_docs, relevant_docs, k))
        
        precision_at_k[k] = sum(precision_scores) / len(precision_scores)
        recall_at_k[k] = sum(recall_scores) / len(recall_scores)
        ndcg_at_k[k] = sum(ndcg_scores) / len(ndcg_scores)
        f1_at_k[k] = sum(f1_scores) / len(f1_scores)
    
    # Calculate MRR and hit rate
    mrr = calculate_mrr(retrieved_docs_list, relevant_docs_list)
    hit_rate = calculate_hit_rate(retrieved_docs_list, relevant_docs_list)
    
    return RetrievalMetrics(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        ndcg_at_k=ndcg_at_k,
        mrr=mrr,
        hit_rate=hit_rate,
        f1_at_k=f1_at_k,
    )