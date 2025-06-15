"""Report generation for benchmark results.

This package provides comprehensive reporting capabilities for benchmark results
including JSON, TXT, HTML, and visualization formats.
"""

from .report_generator import (
    ReportGenerator,
    BenchmarkReport,
    ReportConfig,
    generate_report,
    generate_all_reports,
)

from .formatters import (
    JSONFormatter,
    TextFormatter,
    HTMLFormatter,
    MarkdownFormatter,
)

from .visualizations import (
    create_performance_charts,
    create_comparison_plots,
    create_regression_analysis,
    save_visualization,
)

__all__ = [
    # Core report generation
    "ReportGenerator",
    "BenchmarkReport",
    "ReportConfig",
    "generate_report",
    "generate_all_reports",
    
    # Formatters
    "JSONFormatter",
    "TextFormatter", 
    "HTMLFormatter",
    "MarkdownFormatter",
    
    # Visualizations
    "create_performance_charts",
    "create_comparison_plots",
    "create_regression_analysis",
    "save_visualization",
]