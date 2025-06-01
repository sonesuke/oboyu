"""Unified metrics calculation for all benchmark types."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class IRMetrics:
    """Information Retrieval metrics for accuracy evaluation."""
    
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float
    hit_rate: float
    f1_at_k: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": self.mrr,
            "hit_rate": self.hit_rate,
            "f1_at_k": self.f1_at_k,
        }


@dataclass
class SpeedMetrics:
    """Speed and performance metrics."""
    
    total_time: float
    throughput: float
    mean_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    queries_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_time": self.total_time,
            "throughput": self.throughput,
            "mean_response_time": self.mean_response_time,
            "median_response_time": self.median_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "queries_per_second": self.queries_per_second,
        }


class MetricsCalculator:
    """Unified metrics calculator for all benchmark types."""
    
    @staticmethod
    def calculate_precision_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """Calculate Precision@K.
        
        Args:
            relevant_docs: Set of relevant document IDs
            retrieved_docs: List of retrieved document IDs in rank order
            k: Number of top documents to consider
            
        Returns:
            Precision@K value

        """
        if k <= 0 or not retrieved_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_in_top_k / min(k, len(top_k))
    
    @staticmethod
    def calculate_recall_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """Calculate Recall@K.
        
        Args:
            relevant_docs: Set of relevant document IDs
            retrieved_docs: List of retrieved document IDs in rank order
            k: Number of top documents to consider
            
        Returns:
            Recall@K value

        """
        if not relevant_docs or k <= 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_in_top_k / len(relevant_docs)
    
    @staticmethod
    def calculate_ndcg_at_k(
        relevant_docs: Set[str],
        retrieved_docs: List[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate NDCG@K.
        
        Args:
            relevant_docs: Set of relevant document IDs
            retrieved_docs: List of retrieved document IDs in rank order
            k: Number of top documents to consider
            relevance_scores: Optional relevance scores for documents
            
        Returns:
            NDCG@K value

        """
        if k <= 0 or not retrieved_docs or not relevant_docs:
            return 0.0
        
        # DCG calculation
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            if doc in relevant_docs:
                # Use provided relevance score or default to 1
                relevance = relevance_scores.get(doc, 1.0) if relevance_scores else 1.0
                dcg += relevance / np.log2(i + 2)  # i+2 because rank starts from 1
        
        # IDCG calculation (ideal ranking)
        ideal_scores = []
        for doc in relevant_docs:
            score = relevance_scores.get(doc, 1.0) if relevance_scores else 1.0
            ideal_scores.append(score)
        ideal_scores.sort(reverse=True)
        
        idcg = 0.0
        for i, score in enumerate(ideal_scores[:k]):
            idcg += score / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def calculate_mrr(relevant_docs: Set[str], retrieved_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank.
        
        Args:
            relevant_docs: Set of relevant document IDs
            retrieved_docs: List of retrieved document IDs in rank order
            
        Returns:
            Reciprocal rank (0 if no relevant documents found)

        """
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def calculate_f1_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """Calculate F1@K.
        
        Args:
            relevant_docs: Set of relevant document IDs
            retrieved_docs: List of retrieved document IDs in rank order
            k: Number of top documents to consider
            
        Returns:
            F1@K value

        """
        precision = MetricsCalculator.calculate_precision_at_k(relevant_docs, retrieved_docs, k)
        recall = MetricsCalculator.calculate_recall_at_k(relevant_docs, retrieved_docs, k)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_hit_rate(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """Calculate Hit Rate@K (binary: 1 if any relevant doc in top-k, 0 otherwise).
        
        Args:
            relevant_docs: Set of relevant document IDs
            retrieved_docs: List of retrieved document IDs in rank order
            k: Number of top documents to consider
            
        Returns:
            Hit rate (0.0 or 1.0)

        """
        top_k = retrieved_docs[:k]
        return 1.0 if any(doc in relevant_docs for doc in top_k) else 0.0
    
    @classmethod
    def calculate_ir_metrics(
        self,
        relevant_docs: Set[str],
        retrieved_docs: List[str],
        k_values: List[int] = [1, 5, 10, 20],
        relevance_scores: Optional[Dict[str, float]] = None,
    ) -> IRMetrics:
        """Calculate all IR metrics for a single query.
        
        Args:
            relevant_docs: Set of relevant document IDs
            retrieved_docs: List of retrieved document IDs in rank order
            k_values: List of k values to calculate metrics for
            relevance_scores: Optional relevance scores for documents
            
        Returns:
            IRMetrics object with all calculated metrics

        """
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        f1_at_k = {}
        
        for k in k_values:
            precision_at_k[k] = self.calculate_precision_at_k(relevant_docs, retrieved_docs, k)
            recall_at_k[k] = self.calculate_recall_at_k(relevant_docs, retrieved_docs, k)
            ndcg_at_k[k] = self.calculate_ndcg_at_k(relevant_docs, retrieved_docs, k, relevance_scores)
            f1_at_k[k] = self.calculate_f1_at_k(relevant_docs, retrieved_docs, k)
        
        mrr = self.calculate_mrr(relevant_docs, retrieved_docs)
        hit_rate = self.calculate_hit_rate(relevant_docs, retrieved_docs, max(k_values) if k_values else 10)
        
        return IRMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr,
            hit_rate=hit_rate,
            f1_at_k=f1_at_k,
        )
    
    @classmethod
    def aggregate_ir_metrics(self, metrics_list: List[IRMetrics]) -> IRMetrics:
        """Aggregate IR metrics across multiple queries.
        
        Args:
            metrics_list: List of IRMetrics objects to aggregate
            
        Returns:
            Aggregated IRMetrics object

        """
        if not metrics_list:
            return IRMetrics({}, {}, {}, 0.0, 0.0, {})
        
        # Get all k values
        all_k_values = set()
        for metrics in metrics_list:
            all_k_values.update(metrics.precision_at_k.keys())
        
        # Aggregate metrics
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        f1_at_k = {}
        
        for k in all_k_values:
            precision_values = [m.precision_at_k.get(k, 0.0) for m in metrics_list]
            recall_values = [m.recall_at_k.get(k, 0.0) for m in metrics_list]
            ndcg_values = [m.ndcg_at_k.get(k, 0.0) for m in metrics_list]
            f1_values = [m.f1_at_k.get(k, 0.0) for m in metrics_list]
            
            precision_at_k[k] = sum(precision_values) / len(precision_values)
            recall_at_k[k] = sum(recall_values) / len(recall_values)
            ndcg_at_k[k] = sum(ndcg_values) / len(ndcg_values)
            f1_at_k[k] = sum(f1_values) / len(f1_values)
        
        # Aggregate single-value metrics
        mrr = sum(m.mrr for m in metrics_list) / len(metrics_list)
        hit_rate = sum(m.hit_rate for m in metrics_list) / len(metrics_list)
        
        return IRMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr,
            hit_rate=hit_rate,
            f1_at_k=f1_at_k,
        )
    
    @staticmethod
    def calculate_speed_metrics(response_times: List[float]) -> SpeedMetrics:
        """Calculate speed metrics from response times.
        
        Args:
            response_times: List of response times in seconds
            
        Returns:
            SpeedMetrics object

        """
        if not response_times:
            return SpeedMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        response_times_array = np.array(response_times)
        total_time = sum(response_times)
        
        return SpeedMetrics(
            total_time=total_time,
            throughput=len(response_times) / total_time if total_time > 0 else 0.0,
            mean_response_time=float(np.mean(response_times_array)),
            median_response_time=float(np.median(response_times_array)),
            p95_response_time=float(np.percentile(response_times_array, 95)),
            p99_response_time=float(np.percentile(response_times_array, 99)),
            queries_per_second=len(response_times) / total_time if total_time > 0 else 0.0,
        )
