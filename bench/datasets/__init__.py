"""Dataset management for benchmark evaluation.

This package provides unified dataset management for different types of
evaluation datasets including JMTEB datasets and custom datasets.
"""

from .dataset_manager import (
    DatasetManager,
    DatasetInfo,
    EvaluationDataset,
    Query,
    Document,
    download_dataset,
    load_dataset,
    validate_dataset,
)

from .downloaders import (
    MiraclJADownloader,
    MLDRJADownloader,
    JAGovFAQsDownloader,
    JACWIRDownloader,
    get_downloader,
)

__all__ = [
    # Core dataset management
    "DatasetManager",
    "DatasetInfo", 
    "EvaluationDataset",
    "Query",
    "Document",
    "download_dataset",
    "load_dataset",
    "validate_dataset",
    
    # Dataset downloaders
    "MiraclJADownloader",
    "MLDRJADownloader", 
    "JAGovFAQsDownloader",
    "JACWIRDownloader",
    "get_downloader",
]