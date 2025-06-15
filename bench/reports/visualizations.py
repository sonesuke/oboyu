"""Visualization generation for benchmark reports.

This module provides functions to create charts and graphs for benchmark results.
Note: Visualization libraries (matplotlib, plotly) are optional dependencies.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

from ..core.benchmark_base import BenchmarkResult


logger = logging.getLogger(__name__)


def create_performance_charts(results: List[BenchmarkResult]) -> Optional[Any]:
    """Create performance charts for benchmark results.
    
    Args:
        results: List of benchmark results
        
    Returns:
        Chart object or None if visualization libraries not available
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not results:
            return None
        
        # Extract data for plotting
        benchmark_types = [r.benchmark_type for r in results]
        durations = [r.duration_seconds for r in results]
        success_flags = [r.success for r in results]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Duration chart
        colors = ['green' if success else 'red' for success in success_flags]
        bars = ax1.bar(benchmark_types, durations, color=colors, alpha=0.7)
        ax1.set_title('Benchmark Duration by Type')
        ax1.set_xlabel('Benchmark Type')
        ax1.set_ylabel('Duration (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, duration in zip(bars, durations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{duration:.1f}s',
                    ha='center', va='bottom')
        
        # Success rate pie chart
        success_count = sum(success_flags)
        failure_count = len(results) - success_count
        
        if success_count > 0 or failure_count > 0:
            sizes = [success_count, failure_count]
            labels = ['Success', 'Failure']
            colors_pie = ['green', 'red']
            
            # Only show non-zero slices
            if failure_count == 0:
                sizes = [success_count]
                labels = ['Success']
                colors_pie = ['green']
            elif success_count == 0:
                sizes = [failure_count]
                labels = ['Failure']
                colors_pie = ['red']
            
            ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Success vs Failure Rate')
        
        plt.tight_layout()
        return fig
        
    except ImportError:
        logger.warning("Matplotlib not available - skipping performance charts")
        return None
    except Exception as e:
        logger.error(f"Error creating performance charts: {e}")
        return None


def create_comparison_plots(
    current_results: List[BenchmarkResult],
    previous_results: List[BenchmarkResult],
) -> Optional[Any]:
    """Create comparison plots between current and previous results.
    
    Args:
        current_results: Current benchmark results
        previous_results: Previous benchmark results
        
    Returns:
        Plot object or None if visualization libraries not available
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Group results by benchmark type
        current_by_type = {r.benchmark_type: r for r in current_results if r.success}
        previous_by_type = {r.benchmark_type: r for r in previous_results if r.success}
        
        # Find common benchmark types
        common_types = set(current_by_type.keys()) & set(previous_by_type.keys())
        if not common_types:
            logger.warning("No common benchmark types for comparison")
            return None
        
        # Extract data for comparison
        benchmark_types = list(common_types)
        current_durations = [current_by_type[bt].duration_seconds for bt in benchmark_types]
        previous_durations = [previous_by_type[bt].duration_seconds for bt in benchmark_types]
        
        # Create comparison chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(benchmark_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, previous_durations, width, label='Previous', alpha=0.7)
        bars2 = ax.bar(x + width/2, current_durations, width, label='Current', alpha=0.7)
        
        ax.set_title('Benchmark Duration Comparison')
        ax.set_xlabel('Benchmark Type')
        ax.set_ylabel('Duration (seconds)')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmark_types, rotation=45)
        ax.legend()
        
        # Add value labels on bars
        for bars, durations in [(bars1, previous_durations), (bars2, current_durations)]:
            for bar, duration in zip(bars, durations):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{duration:.1f}s',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
        
    except ImportError:
        logger.warning("Matplotlib not available - skipping comparison plots")
        return None
    except Exception as e:
        logger.error(f"Error creating comparison plots: {e}")
        return None


def create_regression_analysis(
    historical_results: List[List[BenchmarkResult]],
    benchmark_type: str = None,
) -> Optional[Any]:
    """Create regression analysis charts from historical results.
    
    Args:
        historical_results: List of result lists ordered by time
        benchmark_type: Specific benchmark type to analyze (all if None)
        
    Returns:
        Chart object or None if visualization libraries not available
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime
        
        if len(historical_results) < 2:
            logger.warning("Need at least 2 historical result sets for regression analysis")
            return None
        
        # Extract time series data
        if benchmark_type:
            # Analyze specific benchmark type
            durations = []
            timestamps = []
            
            for results in historical_results:
                for result in results:
                    if result.benchmark_type == benchmark_type and result.success:
                        durations.append(result.duration_seconds)
                        timestamps.append(result.timestamp)
                        break
            
            if len(durations) < 2:
                logger.warning(f"Insufficient data for {benchmark_type} regression analysis")
                return None
            
            # Create time series plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(len(durations)), durations, marker='o', linewidth=2, markersize=6)
            ax.set_title(f'Performance Trend: {benchmark_type}')
            ax.set_xlabel('Run Number')
            ax.set_ylabel('Duration (seconds)')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(durations) > 2:
                z = np.polyfit(range(len(durations)), durations, 1)
                p = np.poly1d(z)
                ax.plot(range(len(durations)), p(range(len(durations))), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
                ax.legend()
            
        else:
            # Analyze all benchmark types
            benchmark_types = set()
            for results in historical_results:
                benchmark_types.update(r.benchmark_type for r in results if r.success)
            
            if not benchmark_types:
                logger.warning("No successful results for regression analysis")
                return None
            
            # Create subplot for each benchmark type
            n_types = len(benchmark_types)
            cols = min(3, n_types)
            rows = (n_types + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_types == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if isinstance(axes, list) else [axes]
            else:
                axes = axes.flatten()
            
            for i, bt in enumerate(sorted(benchmark_types)):
                ax = axes[i] if i < len(axes) else None
                if ax is None:
                    continue
                
                # Extract durations for this benchmark type
                durations = []
                for results in historical_results:
                    for result in results:
                        if result.benchmark_type == bt and result.success:
                            durations.append(result.duration_seconds)
                            break
                    else:
                        durations.append(None)  # Missing data point
                
                # Plot only non-None values
                valid_indices = [i for i, d in enumerate(durations) if d is not None]
                valid_durations = [durations[i] for i in valid_indices]
                
                if len(valid_durations) >= 2:
                    ax.plot(valid_indices, valid_durations, marker='o', linewidth=2, markersize=4)
                    ax.set_title(f'{bt}', fontsize=10)
                    ax.set_xlabel('Run')
                    ax.set_ylabel('Duration (s)')
                    ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(benchmark_types), len(axes)):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
        
    except ImportError:
        logger.warning("Matplotlib not available - skipping regression analysis")
        return None
    except Exception as e:
        logger.error(f"Error creating regression analysis: {e}")
        return None


def create_metrics_heatmap(
    results: List[BenchmarkResult],
    metrics_to_plot: List[str] = None,
) -> Optional[Any]:
    """Create a heatmap of metrics across benchmark types.
    
    Args:
        results: Benchmark results
        metrics_to_plot: Specific metrics to include (all if None)
        
    Returns:
        Heatmap object or None if visualization libraries not available
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract metrics data
        benchmark_types = []
        all_metrics = {}
        
        for result in results:
            if not result.success or not result.metrics:
                continue
            
            benchmark_types.append(result.benchmark_type)
            
            # Flatten nested metrics
            flattened_metrics = {}
            for key, value in result.metrics.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            flattened_metrics[f"{key}.{subkey}"] = subvalue
                elif isinstance(value, (int, float)):
                    flattened_metrics[key] = value
            
            all_metrics[result.benchmark_type] = flattened_metrics
        
        if not all_metrics:
            logger.warning("No metrics data available for heatmap")
            return None
        
        # Find common metrics across all benchmarks
        all_metric_names = set()
        for metrics in all_metrics.values():
            all_metric_names.update(metrics.keys())
        
        if metrics_to_plot:
            metric_names = [m for m in metrics_to_plot if m in all_metric_names]
        else:
            metric_names = sorted(all_metric_names)
        
        if not metric_names:
            logger.warning("No valid metrics found for heatmap")
            return None
        
        # Create matrix
        matrix = []
        for bt in benchmark_types:
            row = []
            for metric in metric_names:
                value = all_metrics[bt].get(metric, 0)
                row.append(value)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Normalize each column (metric) to 0-1 scale
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            if col.max() > col.min():
                matrix[:, j] = (col - col.min()) / (col.max() - col.min())
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(metric_names)), max(6, len(benchmark_types))))
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(benchmark_types)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(benchmark_types)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Normalized Value', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(benchmark_types)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if matrix[i, j] < 0.5 else "white")
        
        ax.set_title("Metrics Heatmap Across Benchmark Types")
        plt.tight_layout()
        return fig
        
    except ImportError:
        logger.warning("Matplotlib not available - skipping metrics heatmap")
        return None
    except Exception as e:
        logger.error(f"Error creating metrics heatmap: {e}")
        return None


def save_visualization(figure: Any, filepath: Path) -> bool:
    """Save a visualization figure to file.
    
    Args:
        figure: matplotlib or plotly figure object
        filepath: Path to save the figure
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on figure type
        if hasattr(figure, 'savefig'):  # matplotlib
            figure.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {filepath}")
            return True
        elif hasattr(figure, 'write_image'):  # plotly
            figure.write_image(str(filepath))
            logger.info(f"Saved visualization to {filepath}")
            return True
        else:
            logger.warning(f"Unknown figure type for saving: {type(figure)}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to save visualization to {filepath}: {e}")
        return False


def create_interactive_dashboard(results: List[BenchmarkResult]) -> Optional[str]:
    """Create an interactive dashboard with plotly.
    
    Args:
        results: Benchmark results
        
    Returns:
        HTML string for interactive dashboard or None if not available
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
        
        if not results:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Duration by Type', 'Success Rate', 'Metrics Overview', 'Performance Trend'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Duration bar chart
        benchmark_types = [r.benchmark_type for r in results]
        durations = [r.duration_seconds for r in results]
        success_flags = [r.success for r in results]
        
        colors = ['green' if success else 'red' for success in success_flags]
        
        fig.add_trace(
            go.Bar(x=benchmark_types, y=durations, marker_color=colors, name="Duration"),
            row=1, col=1
        )
        
        # Success rate pie chart
        success_count = sum(success_flags)
        failure_count = len(results) - success_count
        
        fig.add_trace(
            go.Pie(labels=['Success', 'Failure'], values=[success_count, failure_count],
                   marker_colors=['green', 'red']),
            row=1, col=2
        )
        
        # Convert to HTML
        html = pyo.plot(fig, output_type='div', include_plotlyjs=True)
        return html
        
    except ImportError:
        logger.warning("Plotly not available - skipping interactive dashboard")
        return None
    except Exception as e:
        logger.error(f"Error creating interactive dashboard: {e}")
        return None