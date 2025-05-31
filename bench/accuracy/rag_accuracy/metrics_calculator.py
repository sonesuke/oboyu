"""Metrics Calculator for RAG System Evaluation.

This module calculates various metrics for evaluating RAG system performance,
including precision, recall, NDCG, MRR, and other retrieval metrics.
"""

import math
from typing import Any, Dict, List, Set

import numpy as np


class MetricsCalculator:
    """Calculates metrics for RAG system evaluation."""

    def calculate_retrieval_metrics(self, query_results: List[Any], k: int) -> Dict[str, float]:
        """Calculate retrieval metrics for a set of queries.

        Args:
            query_results: List of QueryResult objects
            k: Number of top results to consider

        Returns:
            Dictionary of metric names to values

        """
        metrics = {}

        # Calculate individual metrics
        precisions = []
        recalls = []
        ndcgs = []
        mrrs = []
        hit_rates = []

        for result in query_results:
            retrieved_ids = [doc["doc_id"] for doc in result.retrieved_docs[:k]]
            relevant_ids = set(result.relevant_docs)

            if relevant_ids:  # Only calculate if there are relevant docs
                precision = self._precision_at_k(retrieved_ids, relevant_ids, k)
                recall = self._recall_at_k(retrieved_ids, relevant_ids, k)
                ndcg = self._ndcg_at_k(retrieved_ids, relevant_ids, k)
                mrr = self._mrr(retrieved_ids, relevant_ids)
                hit = self._hit_rate(retrieved_ids, relevant_ids)

                precisions.append(precision)
                recalls.append(recall)
                ndcgs.append(ndcg)
                mrrs.append(mrr)
                hit_rates.append(hit)

        # Calculate mean metrics
        if precisions:  # Ensure we have results
            metrics[f"precision@{k}"] = np.mean(precisions)
            metrics[f"recall@{k}"] = np.mean(recalls)
            metrics[f"ndcg@{k}"] = np.mean(ndcgs)
            metrics["mrr"] = np.mean(mrrs)
            metrics["hit_rate"] = np.mean(hit_rates)
            metrics[f"f1@{k}"] = self._f1_score(metrics[f"precision@{k}"], metrics[f"recall@{k}"])
        else:
            # No valid queries with relevant documents
            metrics[f"precision@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
            metrics["mrr"] = 0.0
            metrics["hit_rate"] = 0.0
            metrics[f"f1@{k}"] = 0.0

        return metrics

    def calculate_reranking_metrics(
        self, original_results: List[Any], reranked_results: List[Any], k: int
    ) -> Dict[str, float]:
        """Calculate metrics for reranking effectiveness.

        Args:
            original_results: Original retrieval results
            reranked_results: Results after reranking
            k: Number of top results to consider

        Returns:
            Dictionary of reranking metrics

        """
        metrics = {}

        # Calculate improvement metrics
        original_metrics = self.calculate_retrieval_metrics(original_results, k)
        reranked_metrics = self.calculate_retrieval_metrics(reranked_results, k)

        # Calculate improvements
        for metric_name in original_metrics:
            improvement_key = f"{metric_name}_improvement"
            if original_metrics[metric_name] > 0:
                improvement = (
                    (reranked_metrics[metric_name] - original_metrics[metric_name]) / original_metrics[metric_name]
                ) * 100
            else:
                improvement = 0.0 if reranked_metrics[metric_name] == 0 else float("inf")
            metrics[improvement_key] = improvement

        # Add absolute metrics
        metrics["original_ndcg"] = original_metrics.get(f"ndcg@{k}", 0.0)
        metrics["reranked_ndcg"] = reranked_metrics.get(f"ndcg@{k}", 0.0)
        metrics["original_mrr"] = original_metrics.get("mrr", 0.0)
        metrics["reranked_mrr"] = reranked_metrics.get("mrr", 0.0)

        # Calculate position-based metrics
        position_improvements = []
        for orig, rerank in zip(original_results, reranked_results):
            orig_positions = self._get_relevant_positions(orig.retrieved_docs, orig.relevant_docs)
            rerank_positions = self._get_relevant_positions(rerank.retrieved_docs, rerank.relevant_docs)

            if orig_positions and rerank_positions:
                # Average position improvement (lower is better)
                avg_orig_pos = np.mean(orig_positions)
                avg_rerank_pos = np.mean(rerank_positions)
                position_improvements.append(avg_orig_pos - avg_rerank_pos)

        if position_improvements:
            metrics["avg_position_improvement"] = np.mean(position_improvements)
        else:
            metrics["avg_position_improvement"] = 0.0

        return metrics

    def _precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate precision at k.

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision@k score

        """
        if k == 0:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)

        return relevant_retrieved / len(retrieved_k) if retrieved_k else 0.0

    def _recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate recall at k.

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            Recall@k score

        """
        if not relevant:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)

        return relevant_retrieved / len(relevant)

    def _ndcg_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate NDCG (Normalized Discounted Cumulative Gain) at k.

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            NDCG@k score

        """
        if not relevant:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                # Binary relevance: 1 if relevant, 0 otherwise
                relevance = 1.0
                dcg += relevance / math.log2(i + 2)  # i+2 because positions start at 1

        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        ideal_retrieved = min(len(relevant), k)
        for i in range(ideal_retrieved):
            idcg += 1.0 / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _mrr(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank.

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs

        Returns:
            MRR score (reciprocal of first relevant result position)

        """
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def _hit_rate(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate hit rate (whether at least one relevant doc was retrieved).

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs

        Returns:
            1.0 if at least one relevant doc was retrieved, 0.0 otherwise

        """
        return 1.0 if any(doc_id in relevant for doc_id in retrieved) else 0.0

    def _f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score.

        Args:
            precision: Precision score
            recall: Recall score

        Returns:
            F1 score

        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _get_relevant_positions(self, retrieved_docs: List[Dict[str, Any]], relevant_ids: List[str]) -> List[int]:
        """Get positions of relevant documents in retrieved results.

        Args:
            retrieved_docs: List of retrieved documents
            relevant_ids: List of relevant document IDs

        Returns:
            List of positions (1-indexed) where relevant documents appear

        """
        positions = []
        relevant_set = set(relevant_ids)

        for i, doc in enumerate(retrieved_docs):
            if doc["doc_id"] in relevant_set:
                positions.append(i + 1)  # 1-indexed position

        return positions

    def calculate_aggregate_metrics(self, all_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate aggregate metrics across multiple evaluation runs.

        Args:
            all_results: List of metric dictionaries from multiple runs

        Returns:
            Aggregated metrics with mean, std, min, max

        """
        if not all_results:
            return {}

        aggregate = {}
        all_metrics = set()

        # Collect all metric names
        for result in all_results:
            all_metrics.update(result.keys())

        # Calculate aggregates for each metric
        for metric in all_metrics:
            values = [result.get(metric, 0.0) for result in all_results if metric in result]

            if values:
                aggregate[f"{metric}_mean"] = np.mean(values)
                aggregate[f"{metric}_std"] = np.std(values)
                aggregate[f"{metric}_min"] = np.min(values)
                aggregate[f"{metric}_max"] = np.max(values)

        return aggregate

