"""Results Analyzer for RAG Accuracy Evaluation.

This module provides analysis and reporting capabilities for RAG evaluation results,
including comparisons, trend analysis, and report generation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bench.logger import BenchmarkLogger


class ResultsAnalyzer:
    """Analyzes and reports on RAG evaluation results."""

    def __init__(self, results_dir: Optional[Path] = None, logger: Optional[BenchmarkLogger] = None) -> None:
        """Initialize results analyzer.

        Args:
            results_dir: Directory containing evaluation results
            logger: Optional logger for output

        """
        self.results_dir = results_dir or Path("bench/results")
        self.logger = logger or BenchmarkLogger()

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze RAG evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Analysis summary with insights

        """
        analysis = {
            "summary": self._generate_summary(results),
            "by_dataset": self._analyze_by_dataset(results),
            "by_search_mode": self._analyze_by_search_mode(results),
            "by_top_k": self._analyze_by_top_k(results),
            "performance_insights": self._generate_insights(results),
        }

        return analysis

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall summary of results.

        Args:
            results: Evaluation results

        Returns:
            Summary statistics

        """
        if not results:
            return {}

        # Collect all metrics
        all_metrics = {}
        total_queries = 0
        total_time = 0

        for result in results:
            metrics = result.get("metrics", {})
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

            total_queries += result.get("total_queries", 0)
            total_time += result.get("evaluation_time", 0)

        # Calculate summary statistics
        summary = {
            "total_evaluations": len(results),
            "total_queries_evaluated": total_queries,
            "total_evaluation_time": total_time,
            "datasets_evaluated": len(set(r.get("dataset_name", "") for r in results)),
            "search_modes_evaluated": len(set(r.get("search_mode", "") for r in results)),
            "metric_summaries": {},
        }

        # Summary for each metric
        for metric_name, values in all_metrics.items():
            if values:
                summary["metric_summaries"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                }

        return summary

    def _analyze_by_dataset(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results grouped by dataset.

        Args:
            results: Evaluation results

        Returns:
            Analysis by dataset

        """
        by_dataset = {}

        for result in results:
            dataset = result.get("dataset_name", "unknown")
            if dataset not in by_dataset:
                by_dataset[dataset] = {"results": [], "best_config": None, "metrics": {}}

            by_dataset[dataset]["results"].append(result)

        # Find best configuration for each dataset
        for dataset, data in by_dataset.items():
            # Determine best by NDCG@10
            best_ndcg = -1
            best_config = None

            for result in data["results"]:
                ndcg = result.get("metrics", {}).get("ndcg@10", 0)
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_config = {
                        "search_mode": result.get("search_mode"),
                        "top_k": result.get("top_k"),
                        "ndcg@10": ndcg,
                    }

            data["best_config"] = best_config

            # Aggregate metrics
            for result in data["results"]:
                for metric, value in result.get("metrics", {}).items():
                    if metric not in data["metrics"]:
                        data["metrics"][metric] = []
                    data["metrics"][metric].append(value)

        return by_dataset

    def _analyze_by_search_mode(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results grouped by search mode.

        Args:
            results: Evaluation results

        Returns:
            Analysis by search mode

        """
        by_mode = {}

        for result in results:
            mode = result.get("search_mode", "unknown")
            if mode not in by_mode:
                by_mode[mode] = {"results": [], "avg_metrics": {}, "avg_response_time": 0}

            by_mode[mode]["results"].append(result)

        # Calculate averages for each mode
        for mode, data in by_mode.items():
            # Collect metrics
            metric_values = {}
            response_times = []

            for result in data["results"]:
                for metric, value in result.get("metrics", {}).items():
                    if metric not in metric_values:
                        metric_values[metric] = []
                    metric_values[metric].append(value)

                response_times.append(result.get("avg_response_time", 0))

            # Calculate averages
            for metric, values in metric_values.items():
                data["avg_metrics"][metric] = np.mean(values) if values else 0

            data["avg_response_time"] = np.mean(response_times) if response_times else 0

        return by_mode

    def _analyze_by_top_k(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results grouped by top_k value.

        Args:
            results: Evaluation results

        Returns:
            Analysis by top_k

        """
        by_k = {}

        for result in results:
            k = result.get("top_k", 0)
            if k not in by_k:
                by_k[k] = {"results": [], "avg_metrics": {}}

            by_k[k]["results"].append(result)

        # Calculate averages for each k
        for k, data in by_k.items():
            metric_values = {}

            for result in data["results"]:
                for metric, value in result.get("metrics", {}).items():
                    if metric not in metric_values:
                        metric_values[metric] = []
                    metric_values[metric].append(value)

            # Calculate averages
            for metric, values in metric_values.items():
                data["avg_metrics"][metric] = np.mean(values) if values else 0

        return by_k

    def _generate_insights(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate performance insights from results.

        Args:
            results: Evaluation results

        Returns:
            List of insight strings

        """
        insights = []

        if not results:
            return insights

        # Find best overall configuration
        best_ndcg = -1
        best_config = None
        for result in results:
            ndcg = result.get("metrics", {}).get("ndcg@10", 0)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_config = result

        if best_config:
            insights.append(
                f"Best configuration: {best_config.get('search_mode')} search with "
                f"top_k={best_config.get('top_k')} on {best_config.get('dataset_name')} "
                f"(NDCG@10: {best_ndcg:.4f})"
            )

        # Compare search modes
        by_mode = self._analyze_by_search_mode(results)
        mode_performance = []
        for mode, data in by_mode.items():
            avg_ndcg = data["avg_metrics"].get("ndcg@10", 0)
            avg_time = data["avg_response_time"]
            mode_performance.append((mode, avg_ndcg, avg_time))

        mode_performance.sort(key=lambda x: x[1], reverse=True)
        if len(mode_performance) > 1:
            best_mode = mode_performance[0]
            insights.append(
                f"Best search mode: {best_mode[0]} (avg NDCG@10: {best_mode[1]:.4f}, "
                f"avg response time: {best_mode[2]:.3f}s)"
            )

        # Analyze k value impact
        by_k = self._analyze_by_top_k(results)
        k_values = sorted(by_k.keys())
        if len(k_values) > 1:
            insights.append(f"Evaluated k values: {k_values}")

            # Check if performance improves with k
            ndcg_by_k = [(k, by_k[k]["avg_metrics"].get("ndcg@10", 0)) for k in k_values]
            if all(ndcg_by_k[i][1] <= ndcg_by_k[i + 1][1] for i in range(len(ndcg_by_k) - 1)):
                insights.append("Performance consistently improves with larger k values")
            elif all(ndcg_by_k[i][1] >= ndcg_by_k[i + 1][1] for i in range(len(ndcg_by_k) - 1)):
                insights.append("Performance decreases with larger k values")

        # Language-specific insights
        japanese_datasets = [r for r in results if "ja" in r.get("dataset_name", "").lower()]
        if japanese_datasets:
            avg_ja_ndcg = np.mean([r.get("metrics", {}).get("ndcg@10", 0) for r in japanese_datasets])
            insights.append(f"Average performance on Japanese datasets: NDCG@10 = {avg_ja_ndcg:.4f}")

        return insights

    def generate_report(self, analysis: Dict[str, Any], output_path: Optional[Path] = None) -> str:
        """Generate a comprehensive report from analysis.

        Args:
            analysis: Analysis results from analyze_results
            output_path: Optional path to save the report

        Returns:
            Formatted report string

        """
        report = []
        report.append("=" * 80)
        report.append("RAG SYSTEM ACCURACY EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary section
        summary = analysis.get("summary", {})
        report.append("EVALUATION SUMMARY")
        report.append("-" * 40)
        report.append(f"Total evaluations: {summary.get('total_evaluations', 0)}")
        report.append(f"Total queries evaluated: {summary.get('total_queries_evaluated', 0)}")
        report.append(f"Total evaluation time: {summary.get('total_evaluation_time', 0):.2f}s")
        report.append(f"Datasets evaluated: {summary.get('datasets_evaluated', 0)}")
        report.append(f"Search modes evaluated: {summary.get('search_modes_evaluated', 0)}")
        report.append("")

        # Metric summaries
        if summary.get("metric_summaries"):
            report.append("METRIC SUMMARIES")
            report.append("-" * 40)
            for metric, stats in summary["metric_summaries"].items():
                report.append(f"\n{metric}:")
                report.append(f"  Mean: {stats['mean']:.4f} (±{stats['std']:.4f})")
                report.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                report.append(f"  Median: {stats['median']:.4f}")
            report.append("")

        # Results by dataset
        by_dataset = analysis.get("by_dataset", {})
        if by_dataset:
            report.append("\nRESULTS BY DATASET")
            report.append("-" * 40)
            for dataset, data in by_dataset.items():
                report.append(f"\n{dataset}:")
                if data.get("best_config"):
                    config = data["best_config"]
                    report.append(
                        f"  Best config: {config['search_mode']} search, "
                        f"k={config['top_k']} (NDCG@10: {config['ndcg@10']:.4f})"
                    )
                report.append(f"  Evaluations: {len(data['results'])}")

        # Results by search mode
        by_mode = analysis.get("by_search_mode", {})
        if by_mode:
            report.append("\n\nRESULTS BY SEARCH MODE")
            report.append("-" * 40)
            for mode, data in by_mode.items():
                report.append(f"\n{mode}:")
                report.append(f"  Evaluations: {len(data['results'])}")
                report.append(f"  Avg response time: {data['avg_response_time']:.3f}s")
                if data.get("avg_metrics"):
                    report.append("  Average metrics:")
                    for metric, value in sorted(data["avg_metrics"].items()):
                        report.append(f"    {metric}: {value:.4f}")

        # Insights
        insights = analysis.get("performance_insights", [])
        if insights:
            report.append("\n\nPERFORMANCE INSIGHTS")
            report.append("-" * 40)
            for i, insight in enumerate(insights, 1):
                report.append(f"{i}. {insight}")

        report.append("\n" + "=" * 80)

        report_text = "\n".join(report)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")
            self.logger.success(f"Report saved to {output_path}")

        return report_text

    def compare_runs(self, run1_path: Path, run2_path: Path, regression_threshold: float = 0.1) -> Dict[str, Any]:
        """Compare two evaluation runs.

        Args:
            run1_path: Path to first run results
            run2_path: Path to second run results
            regression_threshold: Threshold for flagging regressions (e.g., 0.1 = 10%)

        Returns:
            Comparison results

        """
        # Load results
        with open(run1_path, "r", encoding="utf-8") as f:
            run1 = json.load(f)
        with open(run2_path, "r", encoding="utf-8") as f:
            run2 = json.load(f)

        comparison = {
            "run1_path": str(run1_path),
            "run2_path": str(run2_path),
            "improvements": [],
            "regressions": [],
            "stable": [],
        }

        # Compare matching configurations
        for r1 in run1:
            # Find matching configuration in run2
            r2_match = None
            for r2 in run2:
                if (
                    r1.get("dataset_name") == r2.get("dataset_name")
                    and r1.get("search_mode") == r2.get("search_mode")
                    and r1.get("top_k") == r2.get("top_k")
                ):
                    r2_match = r2
                    break

            if r2_match:
                # Compare metrics
                for metric in r1.get("metrics", {}):
                    if metric in r2_match.get("metrics", {}):
                        v1 = r1["metrics"][metric]
                        v2 = r2_match["metrics"][metric]

                        if v1 > 0:
                            change = (v2 - v1) / v1
                            config = f"{r1['dataset_name']}/{r1['search_mode']}/k={r1['top_k']}"

                            if change > regression_threshold:
                                comparison["improvements"].append(
                                    {"config": config, "metric": metric, "change": change, "v1": v1, "v2": v2}
                                )
                            elif change < -regression_threshold:
                                comparison["regressions"].append(
                                    {"config": config, "metric": metric, "change": change, "v1": v1, "v2": v2}
                                )
                            else:
                                comparison["stable"].append(
                                    {"config": config, "metric": metric, "change": change, "v1": v1, "v2": v2}
                                )

        return comparison

