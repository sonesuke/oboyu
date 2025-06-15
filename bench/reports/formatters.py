"""Report formatters for different output formats.

This module provides formatters for converting benchmark reports to various
output formats including JSON, TXT, HTML, and Markdown.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .report_generator import BenchmarkReport


class ReportFormatter(ABC):
    """Abstract base class for report formatters."""
    
    @abstractmethod
    def format(self, report: BenchmarkReport) -> Any:
        """Format a benchmark report.
        
        Args:
            report: Report to format
            
        Returns:
            Formatted report content
        """
        pass


class JSONFormatter(ReportFormatter):
    """JSON report formatter."""
    
    def format(self, report: BenchmarkReport) -> Dict[str, Any]:
        """Format report as JSON-serializable dictionary."""
        return report.to_dict()


class TextFormatter(ReportFormatter):
    """Plain text report formatter."""
    
    def format(self, report: BenchmarkReport) -> str:
        """Format report as plain text."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"BENCHMARK REPORT: {report.title}")
        lines.append("=" * 80)
        lines.append(f"Generated: {report.timestamp}")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        summary = report.summary
        lines.append(f"Total Benchmarks: {summary.get('total_benchmarks', 0)}")
        lines.append(f"Successful: {summary.get('successful_benchmarks', 0)}")
        lines.append(f"Failed: {summary.get('failed_benchmarks', 0)}")
        lines.append(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        lines.append(f"Total Duration: {summary.get('total_duration_seconds', 0):.1f}s")
        lines.append(f"Average Duration: {summary.get('average_duration_seconds', 0):.1f}s")
        lines.append("")
        
        # Individual Results
        lines.append("INDIVIDUAL RESULTS")
        lines.append("-" * 40)
        for result in report.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            lines.append(f"{status} {result.benchmark_type}: {result.duration_seconds:.1f}s")
            
            if not result.success and result.error_message:
                lines.append(f"    Error: {result.error_message}")
            
            if result.success and result.metrics:
                lines.append("    Metrics:")
                for metric_name, value in result.metrics.items():
                    if isinstance(value, dict):
                        lines.append(f"      {metric_name}:")
                        for k, v in value.items():
                            lines.append(f"        {k}: {v}")
                    else:
                        lines.append(f"      {metric_name}: {value}")
            
            lines.append("")
        
        # Comparisons
        if report.comparisons:
            lines.append("COMPARISONS WITH PREVIOUS RUN")
            lines.append("-" * 40)
            for benchmark_type, comparison in report.comparisons.items():
                lines.append(f"{benchmark_type}:")
                
                duration_change = comparison.get("duration_change", 0)
                duration_percent = comparison.get("duration_change_percent", 0)
                
                if duration_change < 0:
                    lines.append(f"  Duration: ⬇️ {abs(duration_change):.1f}s faster ({duration_percent:.1f}%)")
                elif duration_change > 0:
                    lines.append(f"  Duration: ⬆️ {duration_change:.1f}s slower (+{duration_percent:.1f}%)")
                else:
                    lines.append(f"  Duration: ➡️ No change")
                
                metrics_comp = comparison.get("metrics_comparison", {})
                if metrics_comp:
                    lines.append("  Metrics:")
                    for metric_name, metric_comp in metrics_comp.items():
                        change_percent = metric_comp.get("change_percent", 0)
                        improved = metric_comp.get("improved", False)
                        
                        if improved:
                            lines.append(f"    {metric_name}: ⬆️ +{change_percent:.1f}%")
                        elif change_percent < 0:
                            lines.append(f"    {metric_name}: ⬇️ {change_percent:.1f}%")
                        else:
                            lines.append(f"    {metric_name}: ➡️ No change")
                
                lines.append("")
        
        # Metadata
        if report.metadata:
            lines.append("METADATA")
            lines.append("-" * 40)
            for key, value in report.metadata.items():
                lines.append(f"{key}: {value}")
            lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("Report generated by Oboyu Benchmark Suite")
        lines.append("=" * 80)
        
        return "\n".join(lines)


class HTMLFormatter(ReportFormatter):
    """HTML report formatter with optional template support."""
    
    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize HTML formatter.
        
        Args:
            template_dir: Directory containing HTML templates
        """
        self.template_dir = template_dir
    
    def format(self, report: BenchmarkReport) -> str:
        """Format report as HTML."""
        # Try to use template if available
        if self.template_dir:
            template_path = self.template_dir / "benchmark_report.html"
            if template_path.exists():
                return self._format_with_template(report, template_path)
        
        # Fallback to built-in template
        return self._format_builtin(report)
    
    def _format_with_template(self, report: BenchmarkReport, template_path: Path) -> str:
        """Format using external template."""
        try:
            # Simple template substitution - could be enhanced with Jinja2
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
            
            # Replace template variables
            html = template.replace("{{title}}", report.title)
            html = html.replace("{{timestamp}}", report.timestamp)
            html = html.replace("{{summary_json}}", json.dumps(report.summary, indent=2))
            html = html.replace("{{results_json}}", json.dumps([r.__dict__ for r in report.results], indent=2))
            
            return html
            
        except Exception as e:
            # Fallback to built-in template
            return self._format_builtin(report)
    
    def _format_builtin(self, report: BenchmarkReport) -> str:
        """Format using built-in HTML template."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 1.2em; color: #007acc; }}
        .result {{ margin: 15px 0; padding: 15px; border-left: 4px solid #ddd; background: #f9f9f9; }}
        .result.success {{ border-left-color: #28a745; }}
        .result.failure {{ border-left-color: #dc3545; }}
        .status {{ font-weight: bold; }}
        .status.success {{ color: #28a745; }}
        .status.failure {{ color: #dc3545; }}
        .metrics {{ margin-top: 10px; }}
        .metric-item {{ margin: 5px 0; padding: 5px 10px; background: white; border-radius: 3px; }}
        .comparison {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .improvement {{ color: #28a745; }}
        .degradation {{ color: #dc3545; }}
        .no-change {{ color: #6c757d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #007acc; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .footer {{ margin-top: 40px; text-align: center; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report.title}</h1>
        <p><strong>Generated:</strong> {report.timestamp}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric">
                <div class="metric-label">Total Benchmarks</div>
                <div class="metric-value">{report.summary.get('total_benchmarks', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{report.summary.get('success_rate', 0):.1%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Duration</div>
                <div class="metric-value">{report.summary.get('total_duration_seconds', 0):.1f}s</div>
            </div>
            <div class="metric">
                <div class="metric-label">Average Duration</div>
                <div class="metric-value">{report.summary.get('average_duration_seconds', 0):.1f}s</div>
            </div>
        </div>
        
        <h2>Results</h2>"""
        
        # Individual results
        for result in report.results:
            status_class = "success" if result.success else "failure"
            status_text = "PASS" if result.success else "FAIL"
            status_icon = "✅" if result.success else "❌"
            
            html += f"""
        <div class="result {status_class}">
            <div class="status {status_class}">{status_icon} {status_text}: {result.benchmark_type}</div>
            <div><strong>Duration:</strong> {result.duration_seconds:.1f}s</div>"""
            
            if not result.success and result.error_message:
                html += f'<div><strong>Error:</strong> {result.error_message}</div>'
            
            if result.success and result.metrics:
                html += '<div class="metrics"><strong>Metrics:</strong>'
                for metric_name, value in result.metrics.items():
                    if isinstance(value, dict):
                        html += f'<div class="metric-item"><strong>{metric_name}:</strong><ul>'
                        for k, v in value.items():
                            html += f'<li>{k}: {v}</li>'
                        html += '</ul></div>'
                    else:
                        html += f'<div class="metric-item">{metric_name}: {value}</div>'
                html += '</div>'
            
            html += '</div>'
        
        # Comparisons
        if report.comparisons:
            html += '<h2>Comparisons with Previous Run</h2>'
            for benchmark_type, comparison in report.comparisons.items():
                html += f'<div class="comparison"><h3>{benchmark_type}</h3>'
                
                duration_change = comparison.get("duration_change", 0)
                duration_percent = comparison.get("duration_change_percent", 0)
                
                if duration_change < 0:
                    html += f'<div class="improvement">⬇️ Duration: {abs(duration_change):.1f}s faster ({duration_percent:.1f}%)</div>'
                elif duration_change > 0:
                    html += f'<div class="degradation">⬆️ Duration: {duration_change:.1f}s slower (+{duration_percent:.1f}%)</div>'
                else:
                    html += f'<div class="no-change">➡️ Duration: No change</div>'
                
                metrics_comp = comparison.get("metrics_comparison", {})
                if metrics_comp:
                    html += '<h4>Metrics:</h4><ul>'
                    for metric_name, metric_comp in metrics_comp.items():
                        change_percent = metric_comp.get("change_percent", 0)
                        improved = metric_comp.get("improved", False)
                        
                        if improved:
                            html += f'<li class="improvement">{metric_name}: ⬆️ +{change_percent:.1f}%</li>'
                        elif change_percent < 0:
                            html += f'<li class="degradation">{metric_name}: ⬇️ {change_percent:.1f}%</li>'
                        else:
                            html += f'<li class="no-change">{metric_name}: ➡️ No change</li>'
                    html += '</ul>'
                
                html += '</div>'
        
        # Visualizations
        if report.visualizations:
            html += '<h2>Visualizations</h2>'
            for viz_name, viz_path in report.visualizations.items():
                viz_filename = Path(viz_path).name
                html += f'<div><img src="{viz_filename}" alt="{viz_name}" style="max-width: 100%; height: auto; margin: 10px 0;"></div>'
        
        html += """
        <div class="footer">
            <p>Report generated by Oboyu Benchmark Suite</p>
        </div>
    </div>
</body>
</html>"""
        
        return html


class MarkdownFormatter(ReportFormatter):
    """Markdown report formatter."""
    
    def format(self, report: BenchmarkReport) -> str:
        """Format report as Markdown."""
        lines = []
        
        # Header
        lines.append(f"# {report.title}")
        lines.append("")
        lines.append(f"**Generated:** {report.timestamp}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        summary = report.summary
        lines.append(f"- **Total Benchmarks:** {summary.get('total_benchmarks', 0)}")
        lines.append(f"- **Successful:** {summary.get('successful_benchmarks', 0)}")
        lines.append(f"- **Failed:** {summary.get('failed_benchmarks', 0)}")
        lines.append(f"- **Success Rate:** {summary.get('success_rate', 0):.1%}")
        lines.append(f"- **Total Duration:** {summary.get('total_duration_seconds', 0):.1f}s")
        lines.append(f"- **Average Duration:** {summary.get('average_duration_seconds', 0):.1f}s")
        lines.append("")
        
        # Results table
        lines.append("## Results")
        lines.append("")
        lines.append("| Benchmark | Status | Duration | Metrics |")
        lines.append("|-----------|--------|----------|---------|")
        
        for result in report.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            duration = f"{result.duration_seconds:.1f}s"
            
            # Summarize metrics
            metrics_summary = ""
            if result.success and result.metrics:
                metric_items = []
                for metric_name, value in result.metrics.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            metric_items.append(f"{metric_name}.{k}: {v}")
                    else:
                        metric_items.append(f"{metric_name}: {value}")
                metrics_summary = "<br>".join(metric_items[:3])  # Limit to first 3 metrics
                if len(metric_items) > 3:
                    metrics_summary += "<br>..."
            
            lines.append(f"| {result.benchmark_type} | {status} | {duration} | {metrics_summary} |")
        
        lines.append("")
        
        # Comparisons
        if report.comparisons:
            lines.append("## Comparisons with Previous Run")
            lines.append("")
            
            for benchmark_type, comparison in report.comparisons.items():
                lines.append(f"### {benchmark_type}")
                lines.append("")
                
                duration_change = comparison.get("duration_change", 0)
                duration_percent = comparison.get("duration_change_percent", 0)
                
                if duration_change < 0:
                    lines.append(f"**Duration:** ⬇️ {abs(duration_change):.1f}s faster ({duration_percent:.1f}%)")
                elif duration_change > 0:
                    lines.append(f"**Duration:** ⬆️ {duration_change:.1f}s slower (+{duration_percent:.1f}%)")
                else:
                    lines.append(f"**Duration:** ➡️ No change")
                
                metrics_comp = comparison.get("metrics_comparison", {})
                if metrics_comp:
                    lines.append("")
                    lines.append("**Metrics Changes:**")
                    for metric_name, metric_comp in metrics_comp.items():
                        change_percent = metric_comp.get("change_percent", 0)
                        improved = metric_comp.get("improved", False)
                        
                        if improved:
                            lines.append(f"- {metric_name}: ⬆️ +{change_percent:.1f}%")
                        elif change_percent < 0:
                            lines.append(f"- {metric_name}: ⬇️ {change_percent:.1f}%")
                        else:
                            lines.append(f"- {metric_name}: ➡️ No change")
                
                lines.append("")
        
        # Visualizations
        if report.visualizations:
            lines.append("## Visualizations")
            lines.append("")
            for viz_name, viz_path in report.visualizations.items():
                viz_filename = Path(viz_path).name
                lines.append(f"![{viz_name}]({viz_filename})")
                lines.append("")
        
        # Metadata
        if report.metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in report.metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("*Report generated by Oboyu Benchmark Suite*")
        
        return "\n".join(lines)


class CSVFormatter(ReportFormatter):
    """CSV report formatter for tabular data export."""
    
    def format(self, report: BenchmarkReport) -> str:
        """Format report as CSV."""
        lines = []
        
        # Header
        lines.append("benchmark_type,status,duration_seconds,error_message,metrics")
        
        # Data rows
        for result in report.results:
            status = "success" if result.success else "failure"
            error_msg = (result.error_message or "").replace(",", ";").replace("\n", " ")
            
            # Flatten metrics to JSON string
            metrics_json = ""
            if result.metrics:
                metrics_json = json.dumps(result.metrics).replace(",", ";")
            
            lines.append(f"{result.benchmark_type},{status},{result.duration_seconds},{error_msg},{metrics_json}")
        
        return "\n".join(lines)