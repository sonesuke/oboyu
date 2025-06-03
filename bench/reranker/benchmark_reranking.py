#!/usr/bin/env python3
"""Benchmark script for evaluating Oboyu reranker performance.

This script evaluates the effectiveness of the Ruri reranker models
for improving RAG (Retrieval-Augmented Generation) performance.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from bench.accuracy.rag_accuracy.reranker_evaluator import RerankerEvaluator
from bench.logger import BenchmarkLogger
from oboyu.common.types import SearchResult
from oboyu.config.indexer import IndexerConfig, ProcessingConfig
from oboyu.indexer import Indexer
from oboyu.retriever.services.reranker import create_reranker


class OboyuRerankerAdapter:
    """Adapter to use Oboyu reranker with the evaluation framework."""
    
    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-reranker-310m",
        use_onnx: bool = True,
        batch_size: int = 8,
    ) -> None:
        """Initialize the Oboyu reranker adapter.
        
        Args:
            model_name: Reranker model name
            use_onnx: Whether to use ONNX optimization
            batch_size: Batch size for reranking
        
        """
        self.reranker = create_reranker(
            model_name=model_name,
            use_onnx=use_onnx,
            batch_size=batch_size,
        )
        self.model_name = model_name
        self.use_onnx = use_onnx
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank documents using Oboyu reranker.
        
        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Number of documents to return
            
        Returns:
            Reranked documents
        
        """
        # Convert to SearchResult format expected by Oboyu reranker
        search_results = []
        for doc in documents:
            search_results.append(SearchResult(
                chunk_id=doc.get("id", ""),
                path=doc.get("path", ""),
                title=doc.get("title", ""),
                content=doc.get("content", ""),
                chunk_index=doc.get("chunk_index", 0),
                language=doc.get("language", "en"),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {}),
            ))
        
        # Rerank using Oboyu reranker
        reranked_results = self.reranker.rerank(
            query=query,
            results=search_results,
            top_k=top_k,
        )
        
        # Convert back to dictionary format
        reranked_docs = []
        for result in reranked_results:
            reranked_docs.append({
                "id": result.chunk_id,
                "path": result.path,
                "title": result.title,
                "content": result.content,
                "chunk_index": result.chunk_index,
                "language": result.language,
                "score": result.score,
                "metadata": result.metadata,
            })
        
        return reranked_docs


def benchmark_reranking_performance(
    indexer: Indexer,
    test_queries: List[Dict[str, Any]],
    reranker_configs: List[Dict[str, Any]],
    top_k_values: List[int] = [5, 10, 20],
    initial_retrieval_k: int = 60,
    logger: Optional[BenchmarkLogger] = None,
) -> Dict[str, Any]:
    """Benchmark reranking performance on test queries.
    
    Args:
        indexer: Initialized Oboyu indexer
        test_queries: List of test queries with relevance labels
        reranker_configs: List of reranker configurations to test
        top_k_values: List of k values for evaluation
        initial_retrieval_k: Number of candidates to retrieve initially
        logger: Optional logger
        
    Returns:
        Benchmark results
    
    """
    if logger is None:
        logger = BenchmarkLogger()
    
    logger.section("Starting Reranking Performance Benchmark")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "initial_retrieval_k": initial_retrieval_k,
        "top_k_values": top_k_values,
        "num_queries": len(test_queries),
        "reranker_evaluations": {},
    }
    
    # First, perform initial retrieval for all queries
    logger.info(f"Performing initial retrieval with k={initial_retrieval_k}")
    query_results = []
    
    for i, query_data in enumerate(test_queries):
        query_text = query_data["query"]
        relevant_docs = query_data.get("relevant_docs", [])
        
        # Search without reranking
        search_results = indexer.search(
            query=query_text,
            limit=initial_retrieval_k,
            use_reranker=False,  # Disable reranking for initial retrieval
        )
        
        # Convert to format expected by evaluator
        retrieved_docs = []
        for result in search_results:
            retrieved_docs.append({
                "id": result.chunk_id,
                "path": result.path,
                "title": result.title,
                "content": result.content,
                "chunk_index": result.chunk_index,
                "language": result.language,
                "score": result.score,
                "metadata": result.metadata,
            })
        
        query_result = type(
            "QueryResult",
            (),
            {
                "query_id": str(i),
                "query_text": query_text,
                "retrieved_docs": retrieved_docs,
                "relevant_docs": relevant_docs,
                "retrieval_time": 0.0,  # Not measured here
            },
        )
        query_results.append(query_result)
    
    logger.success(f"Initial retrieval complete for {len(query_results)} queries")
    
    # Evaluate baseline (no reranking)
    logger.section("Evaluating Baseline (No Reranking)")
    baseline_evaluator = RerankerEvaluator(reranker=None, logger=logger)
    baseline_results = baseline_evaluator.evaluate_reranking(query_results, top_k_values)
    results["reranker_evaluations"]["baseline"] = baseline_results
    
    # Evaluate each reranker configuration
    for config in reranker_configs:
        config_name = config.get("name", "unnamed")
        logger.section(f"Evaluating Reranker: {config_name}")
        
        # Create reranker
        reranker = OboyuRerankerAdapter(
            model_name=config.get("model_name", "cl-nagoya/ruri-v3-reranker-310m"),
            use_onnx=config.get("use_onnx", True),
            batch_size=config.get("batch_size", 8),
        )
        
        # Evaluate
        evaluator = RerankerEvaluator(reranker=reranker, logger=logger)
        reranker_results = evaluator.evaluate_reranking(query_results, top_k_values)
        
        # Add configuration info
        reranker_results["config"] = config
        results["reranker_evaluations"][config_name] = reranker_results
    
    # Compare rerankers
    logger.section("Reranker Comparison Summary")
    _print_comparison_summary(results, logger)
    
    return results


def _print_comparison_summary(results: Dict[str, Any], logger: BenchmarkLogger) -> None:
    """Print comparison summary of reranker performance."""
    evaluations = results["reranker_evaluations"]
    top_k_values = results["top_k_values"]
    
    # For each metric and k value, show comparison
    metrics_to_compare = ["hit_rate", "mrr", "ndcg", "precision", "recall"]
    
    for k in top_k_values:
        logger.info(f"\nComparison for top_k={k}:")
        
        # Build comparison table
        headers = ["Reranker"] + metrics_to_compare + ["Avg Time (ms)"]
        rows = []
        
        for name, eval_result in evaluations.items():
            if k in eval_result["top_k_evaluations"]:
                k_result = eval_result["top_k_evaluations"][k]
                row = [name]
                
                # Add metrics
                for metric in metrics_to_compare:
                    value = k_result["metrics"].get(metric, 0.0)
                    row.append(f"{value:.4f}")
                
                # Add timing
                avg_time_ms = k_result["timing_stats"]["avg_reranking_time"] * 1000
                row.append(f"{avg_time_ms:.2f}")
                
                rows.append(row)
        
        # Sort by MRR (descending)
        rows.sort(key=lambda x: float(x[2]), reverse=True)
        
        # Print table
        logger.table(headers, rows)
    
    # Print improvement summary
    logger.info("\nImprovement over baseline:")
    baseline_results = evaluations.get("baseline", {})
    
    for name, eval_result in evaluations.items():
        if name == "baseline":
            continue
        
        logger.info(f"\n{name}:")
        for k in top_k_values:
            if k in eval_result["top_k_evaluations"] and k in baseline_results["top_k_evaluations"]:
                k_result = eval_result["top_k_evaluations"][k]
                baseline_k = baseline_results["top_k_evaluations"][k]
                
                improvements = []
                for metric in ["hit_rate", "mrr", "ndcg"]:
                    baseline_val = baseline_k["metrics"].get(metric, 0.0)
                    reranker_val = k_result["metrics"].get(metric, 0.0)
                    if baseline_val > 0:
                        improvement = ((reranker_val - baseline_val) / baseline_val) * 100
                        improvements.append(f"{metric}: +{improvement:.1f}%")
                
                logger.info(f"  k={k}: {', '.join(improvements)}")


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Oboyu reranker performance")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to benchmark configuration file",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to Oboyu database",
    )
    parser.add_argument(
        "--test-queries",
        type=Path,
        required=True,
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save benchmark results",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        help="Top-k values to evaluate",
    )
    parser.add_argument(
        "--initial-k",
        type=int,
        default=60,
        help="Initial retrieval k (should be >= max top-k * 3)",
    )
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = BenchmarkLogger()
    
    # Load test queries
    logger.info(f"Loading test queries from {args.test_queries}")
    with open(args.test_queries, "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    
    # Load reranker configurations
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        reranker_configs = config.get("rerankers", [])
    else:
        # Default configurations
        reranker_configs = [
            {
                "name": "ruri-v3-reranker-310m-onnx",
                "model_name": "cl-nagoya/ruri-v3-reranker-310m",
                "use_onnx": True,
                "batch_size": 8,
            },
            {
                "name": "ruri-v3-reranker-310m-pytorch",
                "model_name": "cl-nagoya/ruri-v3-reranker-310m",
                "use_onnx": False,
                "batch_size": 8,
            },
        ]
    
    # Initialize indexer
    logger.info(f"Initializing indexer with database: {args.db_path}")
    indexer_config = IndexerConfig(
        processing=ProcessingConfig(db_path=args.db_path)
    )
    indexer = Indexer(config=indexer_config)
    
    # Run benchmark
    results = benchmark_reranking_performance(
        indexer=indexer,
        test_queries=test_queries,
        reranker_configs=reranker_configs,
        top_k_values=args.top_k,
        initial_retrieval_k=args.initial_k,
        logger=logger,
    )
    
    # Save results
    if args.output:
        logger.info(f"Saving results to {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    logger.success("Benchmark complete!")
    
    # Print key findings
    logger.section("Key Findings")
    best_performers = {}
    
    for k in args.top_k:
        best_mrr = 0.0
        best_name = "baseline"
        
        for name, eval_result in results["reranker_evaluations"].items():
            if k in eval_result["top_k_evaluations"]:
                mrr = eval_result["top_k_evaluations"][k]["metrics"].get("mrr", 0.0)
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_name = name
        
        best_performers[k] = (best_name, best_mrr)
    
    logger.info("Best performers by MRR:")
    for k, (name, mrr) in best_performers.items():
        logger.info(f"  k={k}: {name} (MRR={mrr:.4f})")


if __name__ == "__main__":
    main()
