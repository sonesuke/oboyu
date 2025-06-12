"""Crawler services package.

This package contains focused, single-responsibility services for document crawling:
- FileDiscoveryService: Finds files that match criteria
- ContentExtractor: Extracts content from different file formats
- LanguageDetector: Detects the language of document content
- EncodingDetector: Handles encoding detection and conversion
- MetadataExtractor: Extracts file metadata
"""

from oboyu.crawler.services.content_extractor import ContentExtractor
from oboyu.crawler.services.encoding_detector import EncodingDetector
from oboyu.crawler.services.file_discovery_service import FileDiscoveryService
from oboyu.crawler.services.language_detector import LanguageDetector
from oboyu.crawler.services.metadata_extractor import MetadataExtractor
from oboyu.crawler.services.optimized_pdf_processor import OptimizedPDFProcessor

__all__ = [
    "FileDiscoveryService",
    "ContentExtractor",
    "LanguageDetector",
    "EncodingDetector",
    "MetadataExtractor",
    "OptimizedPDFProcessor",
]
