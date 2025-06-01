"""Main crawler implementation for Oboyu.

This module provides the Crawler class for discovering and processing documents.
"""

import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from oboyu.crawler.services import (
    ContentExtractor,
    EncodingDetector,
    FileDiscoveryService,
    LanguageDetector,
    MetadataExtractor,
)


@dataclass
class CrawlerResult:
    """Result of a document crawl operation."""

    path: Path
    """Path to the document."""

    title: str
    """Document title, derived from file name or content."""

    content: str
    """Normalized document content."""

    language: str
    """Detected language code."""

    metadata: Dict[str, object]
    """Additional metadata about the document."""


class Crawler:
    """Document crawler for discovering and extracting content from files.
    
    Orchestrates specialized services to handle different aspects of document crawling:
    - FileDiscoveryService: Finds files that match criteria
    - ContentExtractor: Extracts content from different file formats
    - LanguageDetector: Detects the language of document content
    - EncodingDetector: Handles encoding detection and conversion
    - MetadataExtractor: Extracts file metadata
    """

    def __init__(
        self,
        depth: int = 10,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_workers: int = 4,  # Number of worker threads for parallel processing
        respect_gitignore: bool = True,  # Whether to respect .gitignore files
        discovery_service: Optional[FileDiscoveryService] = None,
        content_extractor: Optional[ContentExtractor] = None,
        language_detector: Optional[LanguageDetector] = None,
        encoding_detector: Optional[EncodingDetector] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ) -> None:
        """Initialize the crawler with configuration options.

        Args:
            depth: Maximum directory traversal depth
            include_patterns: File patterns to include (e.g., "*.txt", "*.md")
            exclude_patterns: Patterns to exclude (e.g., "*/node_modules/*")
            max_workers: Maximum number of worker threads for parallel processing
            respect_gitignore: Whether to respect .gitignore files (default: True)
            discovery_service: Optional custom file discovery service
            content_extractor: Optional custom content extractor
            language_detector: Optional custom language detector
            encoding_detector: Optional custom encoding detector
            metadata_extractor: Optional custom metadata extractor

        Note:
            max_file_size is hard-coded to 10MB and follow_symlinks is hard-coded to False
            for consistency and security reasons.

        """
        self.depth = depth
        self.include_patterns = include_patterns or ["*.txt", "*.md", "*.html", "*.py", "*.java"]
        self.exclude_patterns = exclude_patterns or ["*/node_modules/*", "*/venv/*"]
        self.max_workers = max_workers
        self.respect_gitignore = respect_gitignore

        # Initialize services with defaults if not provided
        self.discovery_service = discovery_service or FileDiscoveryService()
        self.content_extractor = content_extractor or ContentExtractor()
        self.language_detector = language_detector or LanguageDetector()
        self.encoding_detector = encoding_detector or EncodingDetector()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()

        # Keep track of processed files to avoid duplicates
        self._processed_files: Set[Path] = set()

    def crawl(self, directory: Path, progress_callback: Optional[Callable[[str, int, int], None]] = None) -> List[CrawlerResult]:
        """Crawl a directory for documents.

        Args:
            directory: Directory path to crawl
            progress_callback: Optional callback for progress updates (stage, current, total)

        Returns:
            List of processed document results

        """
        # Discover document paths using the discovery service
        doc_paths = self.discovery_service.discover_files(
            root_paths=[directory],
            include_patterns=self.include_patterns,
            exclude_patterns=self.exclude_patterns,
            max_depth=self.depth,
            respect_gitignore=self.respect_gitignore,
        )

        # Filter out already processed files
        new_docs = []
        for doc_path, doc_metadata in doc_paths:
            if doc_path not in self._processed_files:
                new_docs.append((doc_path, doc_metadata))
                self._processed_files.add(doc_path)

        # No new documents to process
        if not new_docs:
            return []

        # Process documents in parallel using ThreadPoolExecutor
        results: List[CrawlerResult] = []
        total_docs = len(new_docs)
        completed_docs = 0

        # Report initial progress
        if progress_callback:
            progress_callback("crawling", 0, total_docs)

        import threading
        import time

        last_progress_time = time.time()

        # Flag to stop periodic updates
        stop_periodic_updates = threading.Event()

        # Function to send periodic progress updates even during long processing
        def periodic_progress_updater() -> None:
            while not stop_periodic_updates.wait(3.0):  # Update every 3 seconds
                if progress_callback:
                    progress_callback("crawling", completed_docs, total_docs)

        # Start periodic updater in background
        update_thread = threading.Thread(target=periodic_progress_updater, daemon=True)
        update_thread.start()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all document processing tasks
                future_to_doc = {
                    executor.submit(self._process_document, doc_path, doc_metadata): (doc_path, doc_metadata) for doc_path, doc_metadata in new_docs
                }

                # Collect results as they complete with periodic progress updates
                for future in concurrent.futures.as_completed(future_to_doc):
                    doc_path, _ = future_to_doc[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        # Log the error and continue
                        print(f"Error processing {doc_path}: {e}")

                    # Report progress
                    completed_docs += 1
                    current_time = time.time()

                    # Report more frequently: every completion or every 2 seconds
                    should_report = (
                        completed_docs == 1
                        or current_time - last_progress_time >= 2.0
                        or completed_docs % 10 == 0
                        or completed_docs == total_docs
                    )
                    if progress_callback and should_report:
                        progress_callback("crawling", completed_docs, total_docs)
                        last_progress_time = current_time
        finally:
            # Stop the periodic updater
            stop_periodic_updates.set()
            update_thread.join(timeout=1.0)
            
            # Always send final progress update to ensure completion
            if progress_callback and completed_docs > 0:
                progress_callback("crawling", completed_docs, total_docs)

        return results

    def _process_document(self, doc_path: Path, doc_metadata: Dict[str, object]) -> Optional[CrawlerResult]:
        """Process a single document.

        Args:
            doc_path: Path to the document
            doc_metadata: Metadata for the document

        Returns:
            CrawlerResult if successful, None otherwise

        """
        try:
            # Extract content using the content extractor service
            content, extracted_metadata = self.content_extractor.extract_content(doc_path)
            
            # Detect language using the language detector service
            language = self.language_detector.detect_language(content)

            # Apply special processing for Japanese text using encoding detector
            if language == "ja":
                encoding = self.encoding_detector.detect_encoding(content)
                content = self.encoding_detector.process_japanese_text(content, encoding)

            # Merge extracted metadata with doc_metadata
            merged_metadata = {**doc_metadata, **extracted_metadata}

            # Generate a title from metadata, filename, or content
            title = self._generate_title(doc_path, content, extracted_metadata)

            # Create the result
            return CrawlerResult(
                path=doc_path,
                title=title,
                content=content,
                language=language,
                metadata=merged_metadata,
            )
        except Exception:
            # The exception will be caught and logged in the calling function
            raise

    def _generate_title(self, path: Path, content: str, metadata: Dict[str, object]) -> str:
        """Generate a title for the document.

        Args:
            path: Path to the document
            content: Document content
            metadata: Extracted metadata from the document

        Returns:
            Generated title

        """
        # First, check if title is in metadata
        if "title" in metadata and metadata["title"]:
            return str(metadata["title"])

        # Try to extract a title from the first line of content
        if content and content.strip():
            first_line = content.strip().splitlines()[0].strip()
            # If first line looks like a title (not too long, no file extension)
            if len(first_line) < 100 and "." not in first_line:
                return first_line

        # Fall back to filename without extension
        return path.stem
