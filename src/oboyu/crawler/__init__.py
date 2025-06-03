"""Document crawler module for Oboyu.

This module is responsible for discovering, extracting, and normalizing documents
from the local file system with specialized handling for Japanese content.
"""

from oboyu.crawler.config import CrawlerConfig
from oboyu.crawler.crawler import Crawler, CrawlerResult

__all__ = [
    "Crawler",
    "CrawlerResult",
    "CrawlerConfig",
]
