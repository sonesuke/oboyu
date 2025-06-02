"""Document crawler module for Oboyu.

This module is responsible for discovering, extracting, and normalizing documents
from the local file system with specialized handling for Japanese content.
"""

from oboyu.crawler.config import CrawlerConfig, load_config_from_file, load_default_config
from oboyu.crawler.crawler import Crawler, CrawlerResult
from oboyu.crawler.discovery import discover_documents
from oboyu.crawler.extractor import extract_content

__all__ = [
    "Crawler",
    "CrawlerResult",
    "CrawlerConfig",
    "discover_documents",
    "extract_content",
    "load_default_config",
    "load_config_from_file",
]
