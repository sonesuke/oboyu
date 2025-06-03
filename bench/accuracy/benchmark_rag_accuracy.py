"""RAG Accuracy Benchmark for Oboyu.

This script runs comprehensive accuracy evaluation for Oboyu as a RAG system,
with focus on Japanese document search performance.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add bench directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag_accuracy import (
    DatasetManager,
    RAGEvaluationConfig,
    RAGEvaluator,
    RerankerEvaluator,
    ResultsAnalyzer,
)

from bench.config import BENCHMARK_CONFIG
from bench.logger import BenchmarkLogger

# Remove utils import - we'll handle paths directly
from oboyu.config.indexer import IndexerConfig, ModelConfig, ProcessingConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments

    """
    parser = argparse.ArgumentParser(description="Run RAG accuracy benchmarks for Oboyu")

    # Dataset options
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["synthetic", "miracl-ja", "mldr-ja", "jagovfaqs-22k", "jacwir", "custom"],
        default=["synthetic"],
        help="Datasets to evaluate on",
    )
    parser.add_argument("--custom-dataset-path", type=Path, help="Path to custom dataset JSON file")

    # Evaluation options
    parser.add_argument(
        "--search-modes",
        nargs="+",
        choices=["vector", "bm25", "hybrid"],
        default=["vector", "bm25", "hybrid"],
        help="Search modes to evaluate",
    )
    parser.add_argument(
        "--top-k-values",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
        help="Top-k values to evaluate",
    )
    parser.add_argument("--test-size", type=int, help="Maximum number of queries to evaluate per dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for query processing")

    # Reranking options
    parser.add_argument("--evaluate-reranking", action="store_true", help="Include reranking evaluation")

    # Output options
    parser.add_argument("--results-dir", type=Path, default=Path("bench/results"), help="Directory to save results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--report-only", action="store_true", help="Only generate report from existing results")

    # Analysis options
    parser.add_argument("--compare-with", type=Path, help="Compare with previous run results")
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.1,
        help="Threshold for regression detection (0.1 = 10%%)",
    )

    return parser.parse_args()


def run_dataset_evaluation(
    evaluator: RAGEvaluator,
    dataset_manager: DatasetManager,
    dataset_name: str,
    custom_path: Optional[Path] = None,
    reindex: bool = True,
) -> List[dict]:
    """Run evaluation on a single dataset.

    Args:
        evaluator: RAG evaluator instance
        dataset_manager: Dataset manager instance
        dataset_name: Name of the dataset
        custom_path: Path for custom dataset
        reindex: Whether to reindex documents

    Returns:
        List of evaluation results

    """
    # Load dataset
    dataset = dataset_manager.load_dataset(dataset_name, custom_path)

    # Prepare for evaluation
    queries, documents = dataset_manager.prepare_dataset_for_evaluation(
        dataset, max_queries=evaluator.config.test_size
    )

    # Run evaluation
    results = evaluator.evaluate_dataset(dataset.name, queries, documents, reindex=reindex)

    return results


def main() -> None:
    """Execute main benchmark."""
    args = parse_args()
    logger = BenchmarkLogger(verbose=args.verbose)

    logger.section("RAG Accuracy Benchmark for Oboyu")
    logger.info(f"Started at: {datetime.now()}")

    # Create results directory
    args.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If report-only mode, generate report and exit
    if args.report_only:
        logger.section("Generating Report from Existing Results")
        analyzer = ResultsAnalyzer(args.results_dir, logger)

        # Find latest results file
        result_files = list(args.results_dir.glob("rag_accuracy_*.json"))
        if not result_files:
            logger.error("No results files found in results directory")
            return

        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using results from: {latest_file}")

        with open(latest_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        analysis = analyzer.analyze_results(results)
        report_path = args.results_dir / f"rag_report_{timestamp}.txt"
        report = analyzer.generate_report(analysis, report_path)
        print("\n" + report)
        return

    # Setup evaluation configuration
    db_path = args.results_dir / "rag_accuracy_test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean up any existing test database to ensure clean state
    if db_path.exists():
        db_path.unlink()
        logger.info(f"Cleaned up existing test database: {db_path}")

    indexer_config = IndexerConfig(
        processing=ProcessingConfig(
            db_path=db_path,
            chunk_size=BENCHMARK_CONFIG.get("indexing", {}).get("chunk_size", 300),
            chunk_overlap=BENCHMARK_CONFIG.get("indexing", {}).get("chunk_overlap", 75)
        ),
        model=ModelConfig()
    )

    eval_config = RAGEvaluationConfig(
        db_path=str(db_path),
        indexer_config=indexer_config,
        top_k_values=args.top_k_values,
        search_modes=args.search_modes,
        batch_size=args.batch_size,
        test_size=args.test_size,
        verbose=args.verbose,
    )

    # Initialize components
    dataset_manager = DatasetManager(logger)
    evaluator = RAGEvaluator(eval_config, logger)
    analyzer = ResultsAnalyzer(args.results_dir, logger)

    all_results = []

    # Run evaluation for each dataset
    for i, dataset_name in enumerate(args.datasets):
        logger.section(f"Dataset {i+1}/{len(args.datasets)}: {dataset_name}")

        try:
            # Recreate evaluator for each dataset to ensure clean state
            if i > 0:
                # Close previous evaluator if it has a db attribute
                if hasattr(evaluator, 'db'):
                    evaluator.db.close()
                
                # Clean up database
                if db_path.exists():
                    db_path.unlink()
                    logger.info("Cleaned up database for fresh start")
                
                # Recreate evaluator
                evaluator = RAGEvaluator(eval_config, logger)
            
            custom_path = args.custom_dataset_path if dataset_name == "custom" else None
            results = run_dataset_evaluation(
                evaluator,
                dataset_manager,
                dataset_name,
                custom_path,
                reindex=True,  # Always reindex for fair comparison
            )
            all_results.extend(results)

            # Optionally run reranking evaluation
            if args.evaluate_reranking and results:
                logger.section("Evaluating Reranking")
                reranker_evaluator = RerankerEvaluator(logger=logger)

                # Get query results from first configuration for reranking eval
                sample_result = results[0]
                if "query_results" in sample_result:
                    rerank_eval = reranker_evaluator.evaluate_reranking(
                        sample_result["query_results"], args.top_k_values
                    )

                    # Save reranking results
                    rerank_path = args.results_dir / f"rag_reranking_{dataset_name}_{timestamp}.json"
                    rerank_path.write_text(json.dumps(rerank_eval, indent=2, ensure_ascii=False))

                    # Print reranking report
                    rerank_report = reranker_evaluator.generate_report(rerank_eval)
                    print("\n" + rerank_report)

        except Exception as e:
            logger.error(f"Failed to evaluate {dataset_name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            # Continue with next dataset instead of stopping
            continue

    # Save all results
    if all_results:
        results_path = args.results_dir / f"rag_accuracy_{timestamp}.json"
        evaluator.save_results(all_results, str(results_path))

        # Generate analysis and report
        logger.section("Analyzing Results")
        # Load the saved results for analysis (they're properly serialized)
        with open(results_path, "r", encoding="utf-8") as f:
            saved_results = json.load(f)
        analysis = analyzer.analyze_results(saved_results)

        report_path = args.results_dir / f"rag_report_{timestamp}.txt"
        report = analyzer.generate_report(analysis, report_path)
        print("\n" + report)

        # Compare with previous run if specified
        if args.compare_with and args.compare_with.exists():
            logger.section("Comparing with Previous Run")
            comparison = analyzer.compare_runs(args.compare_with, results_path, args.regression_threshold)

            # Report comparison results
            logger.info(f"\nComparison: {args.compare_with.name} vs {results_path.name}")
            logger.info(f"Improvements: {len(comparison['improvements'])}")
            logger.info(f"Regressions: {len(comparison['regressions'])}")
            logger.info(f"Stable: {len(comparison['stable'])}")

            if comparison["regressions"]:
                logger.warning("\nRegressions detected:")
                for reg in comparison["regressions"]:
                    logger.warning(
                        f"  {reg['config']} - {reg['metric']}: "
                        f"{reg['v1']:.4f} -> {reg['v2']:.4f} ({reg['change']*100:+.1f}%)"
                    )

            if comparison["improvements"]:
                logger.success("\nImprovements:")
                for imp in comparison["improvements"]:
                    logger.success(
                        f"  {imp['config']} - {imp['metric']}: "
                        f"{imp['v1']:.4f} -> {imp['v2']:.4f} ({imp['change']*100:+.1f}%)"
                    )

            # Save comparison results
            comparison_path = args.results_dir / f"rag_comparison_{timestamp}.json"
            comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))

    # Final cleanup
    try:
        if hasattr(evaluator, 'db'):
            evaluator.db.close()
    except:
        pass
    
    # Clean up test database
    if db_path.exists():
        db_path.unlink()
        logger.info("Cleaned up test database")

    logger.success(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main()

