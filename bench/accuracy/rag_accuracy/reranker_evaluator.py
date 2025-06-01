"""Reranker Evaluator for Oboyu RAG System.

This module provides evaluation framework for the planned reranking feature,
measuring how well reranking improves initial retrieval results.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from bench.logger import BenchmarkLogger

from .metrics_calculator import MetricsCalculator


@dataclass
class RerankingResult:
    """Result of reranking evaluation."""

    query_id: str
    original_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    reranking_time: float
    relevant_docs: List[str]


class Reranker(Protocol):
    """Protocol for reranker implementations."""

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank documents for a query.

        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Number of documents to return

        Returns:
            Reranked documents

        """
        ...


class DummyReranker:
    """Dummy reranker for testing the evaluation framework."""

    def __init__(self, improvement_factor: float = 0.2) -> None:
        """Initialize dummy reranker.

        Args:
            improvement_factor: Simulated improvement factor (0-1)

        """
        self.improvement_factor = improvement_factor

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Simulate reranking by boosting relevant documents.

        This is a placeholder implementation that simulates reranking
        by adjusting scores based on title/content matching.
        """
        # Simple simulation: boost documents that contain query terms
        query_terms = set(query.lower().split())
        reranked = []

        for doc in documents:
            # Simulate relevance scoring
            title_terms = set(doc.get("title", "").lower().split())
            content_terms = set(doc.get("content", "").lower().split()[:50])  # First 50 words

            title_overlap = len(query_terms & title_terms)
            content_overlap = len(query_terms & content_terms)

            # Boost score based on term overlap
            boost = (title_overlap * 2 + content_overlap) * self.improvement_factor
            new_score = doc.get("score", 0.5) + boost

            reranked_doc = doc.copy()
            reranked_doc["score"] = min(1.0, new_score)  # Cap at 1.0
            reranked_doc["reranking_boost"] = boost
            reranked.append(reranked_doc)

        # Sort by new score and return top_k
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]


class RerankerEvaluator:
    """Evaluates reranking effectiveness for RAG system."""

    def __init__(self, reranker: Optional[Reranker] = None, logger: Optional[BenchmarkLogger] = None) -> None:
        """Initialize reranker evaluator.

        Args:
            reranker: Reranker implementation (uses dummy if None)
            logger: Optional logger for output

        """
        self.reranker = reranker or DummyReranker()
        self.logger = logger or BenchmarkLogger()
        self.metrics_calculator = MetricsCalculator()

    def evaluate_reranking(
        self, query_results: List[Any], top_k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Any]:
        """Evaluate reranking effectiveness on query results.

        Args:
            query_results: List of QueryResult objects from initial retrieval
            top_k_values: List of k values to evaluate

        Returns:
            Dictionary containing reranking evaluation results

        """
        self.logger.section("Evaluating Reranking Effectiveness")

        evaluation_results = {
            "reranker_type": type(self.reranker).__name__,
            "total_queries": len(query_results),
            "top_k_evaluations": {},
            "aggregate_metrics": {},
            "timing_stats": {},
        }

        # Evaluate for each top_k value
        for k in top_k_values:
            self.logger.info(f"Evaluating reranking for top_k={k}")
            k_results = self._evaluate_for_k(query_results, k)
            evaluation_results["top_k_evaluations"][k] = k_results

        # Calculate aggregate metrics across all k values
        all_improvements = []
        for k_result in evaluation_results["top_k_evaluations"].values():
            if "metrics" in k_result:
                for metric, value in k_result["metrics"].items():
                    if "_improvement" in metric:
                        all_improvements.append(value)

        if all_improvements:
            evaluation_results["aggregate_metrics"]["avg_improvement"] = sum(all_improvements) / len(all_improvements)
            evaluation_results["aggregate_metrics"]["max_improvement"] = max(all_improvements)
            evaluation_results["aggregate_metrics"]["min_improvement"] = min(all_improvements)

        return evaluation_results

    def _evaluate_for_k(self, query_results: List[Any], k: int) -> Dict[str, Any]:
        """Evaluate reranking for a specific k value.

        Args:
            query_results: Original query results
            k: Number of top results to consider

        Returns:
            Evaluation results for this k value

        """
        reranking_results = []
        reranking_times = []

        # Process each query
        for result in query_results:
            # Get top k documents from original results
            original_docs = result.retrieved_docs[:k]

            # Rerank documents
            start_time = time.time()
            reranked_docs = self.reranker.rerank(result.query_text, original_docs, k)
            reranking_time = time.time() - start_time

            reranking_results.append(
                RerankingResult(
                    query_id=result.query_id,
                    original_results=original_docs,
                    reranked_results=reranked_docs,
                    reranking_time=reranking_time,
                    relevant_docs=result.relevant_docs,
                )
            )
            reranking_times.append(reranking_time)

        # Calculate metrics
        metrics = self._calculate_reranking_metrics(reranking_results, k)

        # Calculate timing statistics
        timing_stats = {
            "avg_reranking_time": sum(reranking_times) / len(reranking_times) if reranking_times else 0,
            "total_reranking_time": sum(reranking_times),
            "min_reranking_time": min(reranking_times) if reranking_times else 0,
            "max_reranking_time": max(reranking_times) if reranking_times else 0,
        }

        return {
            "k": k,
            "metrics": metrics,
            "timing_stats": timing_stats,
            "num_queries": len(reranking_results),
        }

    def _calculate_reranking_metrics(self, reranking_results: List[RerankingResult], k: int) -> Dict[str, float]:
        """Calculate reranking effectiveness metrics.

        Args:
            reranking_results: List of reranking results
            k: Number of top results considered

        Returns:
            Dictionary of metrics

        """
        # Convert to format expected by metrics calculator
        original_query_results = []
        reranked_query_results = []

        for result in reranking_results:
            # Create mock query result for original
            original_mock = type(
                "QueryResult",
                (),
                {
                    "query_id": result.query_id,
                    "retrieved_docs": result.original_results,
                    "relevant_docs": result.relevant_docs,
                },
            )
            original_query_results.append(original_mock)

            # Create mock query result for reranked
            reranked_mock = type(
                "QueryResult",
                (),
                {
                    "query_id": result.query_id,
                    "retrieved_docs": result.reranked_results,
                    "relevant_docs": result.relevant_docs,
                },
            )
            reranked_query_results.append(reranked_mock)

        # Calculate metrics using the metrics calculator
        return self.metrics_calculator.calculate_reranking_metrics(original_query_results, reranked_query_results, k)

    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a human-readable report of reranking evaluation.

        Args:
            evaluation_results: Results from evaluate_reranking

        Returns:
            Formatted report string

        """
        report = []
        report.append("=" * 60)
        report.append("RERANKING EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Reranker: {evaluation_results['reranker_type']}")
        report.append(f"Total Queries: {evaluation_results['total_queries']}")
        report.append("")

        # Report for each k value
        for k, k_results in evaluation_results["top_k_evaluations"].items():
            report.append(f"\n--- Results for top_k={k} ---")
            report.append(f"Number of queries: {k_results['num_queries']}")

            # Metrics
            report.append("\nMetrics:")
            metrics = k_results["metrics"]
            for metric, value in sorted(metrics.items()):
                if "_improvement" in metric:
                    sign = "+" if value > 0 else ""
                    report.append(f"  {metric}: {sign}{value:.2f}%")
                else:
                    report.append(f"  {metric}: {value:.4f}")

            # Timing
            report.append("\nTiming Statistics:")
            timing = k_results["timing_stats"]
            report.append(f"  Average reranking time: {timing['avg_reranking_time']*1000:.2f}ms")
            report.append(f"  Min reranking time: {timing['min_reranking_time']*1000:.2f}ms")
            report.append(f"  Max reranking time: {timing['max_reranking_time']*1000:.2f}ms")

        # Aggregate metrics
        if evaluation_results["aggregate_metrics"]:
            report.append("\n\n--- AGGREGATE METRICS ---")
            agg = evaluation_results["aggregate_metrics"]
            report.append(f"Average improvement across all metrics: {agg['avg_improvement']:.2f}%")
            report.append(f"Maximum improvement: {agg['max_improvement']:.2f}%")
            report.append(f"Minimum improvement: {agg['min_improvement']:.2f}%")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

