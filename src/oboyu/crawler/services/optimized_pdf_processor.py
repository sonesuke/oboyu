"""Optimized PDF processor with parallel processing, streaming, and caching."""

import asyncio
import hashlib
import pickle
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    from oboyu.cli.hierarchical_logger import HierarchicalLogger

try:
    import pymupdf
    import pymupdf4llm

    HAS_PYMUPDF = True
except ImportError as e:
    pymupdf = None
    pymupdf4llm = None
    HAS_PYMUPDF = False
    _IMPORT_ERROR = str(e)


class ProcessingStrategy(Enum):
    """PDF processing strategies based on file characteristics."""

    LIGHTWEIGHT = "lightweight"  # <5MB, <50 pages
    STANDARD = "standard"  # 5-15MB, 50-100 pages
    PARALLEL = "parallel"  # 15-30MB, 100-200 pages
    STREAMING = "streaming"  # >30MB, >200 pages


@dataclass
class PDFMetrics:
    """PDF file metrics for processing strategy decision."""

    file_size_mb: float
    total_pages: int
    estimated_processing_time: float
    recommended_strategy: ProcessingStrategy

    @classmethod
    def analyze_pdf(cls, file_path: Path) -> "PDFMetrics":
        """Analyze PDF file to determine optimal processing strategy."""
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        # Quick page count estimation (without full parsing)
        try:
            if HAS_PYMUPDF:
                doc = pymupdf.open(file_path)
                total_pages = doc.page_count
                doc.close()
            else:
                # Fallback estimation based on file size when PyMuPDF unavailable
                total_pages = max(1, int(file_size_mb * 10))  # Rough estimate
        except Exception:
            # Fallback estimation based on file size
            total_pages = max(1, int(file_size_mb * 10))  # Rough estimate

        # Estimate processing time based on test results
        # PyMuPDF4LLM is faster, so adjust estimates
        estimated_time = (file_size_mb * 0.35) + (total_pages * 0.05)

        # Determine strategy
        if file_size_mb < 5 and total_pages < 50:
            strategy = ProcessingStrategy.LIGHTWEIGHT
        elif file_size_mb < 15 and total_pages < 100:
            strategy = ProcessingStrategy.STANDARD
        elif file_size_mb < 30 and total_pages < 200:
            strategy = ProcessingStrategy.PARALLEL
        else:
            strategy = ProcessingStrategy.STREAMING

        return cls(file_size_mb, total_pages, estimated_time, strategy)


class PDFCache:
    """Intelligent caching system for PDF processing results."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize PDF cache.

        Args:
            cache_dir: Optional cache directory path

        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "oboyu_pdf_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key from file content hash and metadata."""
        stat = file_path.stat()

        # Use file size + mtime for quick cache key
        quick_key = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"

        # For large files, use quick key to avoid expensive hashing
        if stat.st_size > 50 * 1024 * 1024:  # >50MB
            return hashlib.md5(quick_key.encode()).hexdigest()  # noqa: S324

        # For smaller files, use content hash for accuracy
        with open(file_path, "rb") as f:
            content_hash = hashlib.md5(f.read()).hexdigest()  # noqa: S324

        return f"{content_hash}_{stat.st_size}"

    def get(self, file_path: Path) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get cached processing result."""
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)  # noqa: S301
            except Exception:
                # Remove corrupted cache
                cache_file.unlink(missing_ok=True)

        return None

    def set(self, file_path: Path, content: str, metadata: Dict[str, Any]) -> None:
        """Cache processing result."""
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump((content, metadata), f)
        except Exception:  # noqa: S110
            # Ignore cache write errors
            pass

    def clear_old_cache(self, max_age_days: int = 30) -> None:
        """Clear cache entries older than specified days."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink(missing_ok=True)


class OptimizedPDFProcessor:
    """Optimized PDF processor with adaptive strategies."""

    def __init__(self, max_file_size: int = 50 * 1024 * 1024, max_workers: int = 4) -> None:
        """Initialize optimized PDF processor.

        Args:
            max_file_size: Maximum file size to process
            max_workers: Maximum worker threads for parallel processing

        """
        self.max_file_size = max_file_size
        self.max_workers = max_workers
        self.cache = PDFCache()

    def extract_pdf(self, file_path: Path, logger: Optional["HierarchicalLogger"] = None) -> Tuple[str, Dict[str, Any]]:
        """Extract content and metadata from PDF file.

        Args:
            file_path: Path to the PDF file
            logger: Optional HierarchicalLogger for progress display

        Returns:
            Tuple of (content, metadata)

        """
        if not HAS_PYMUPDF:
            raise RuntimeError(
                "PyMuPDF4LLM library is required for PDF processing. "
                "Install it with: pip install 'pymupdf4llm>=0.0.25' or uv add pymupdf4llm. "
                f"Import error: {_IMPORT_ERROR if '_IMPORT_ERROR' in globals() else 'Unknown import error'}"
            )

        # Check cache first
        cached_result = self.cache.get(file_path)
        if cached_result:
            # Skip individual cache messages for cleaner aggregate display
            return cached_result

        # Analyze file characteristics
        metrics = PDFMetrics.analyze_pdf(file_path)
        # Don't create individual file operations for cleaner aggregate display

        # Check file size limit
        if file_path.stat().st_size > self.max_file_size:
            raise RuntimeError(f"PDF file too large ({metrics.file_size_mb:.1f}MB). Maximum supported size: {self.max_file_size / (1024 * 1024):.1f}MB")

        # Select processing strategy
        start_time = time.time()

        if metrics.recommended_strategy == ProcessingStrategy.LIGHTWEIGHT:
            content, metadata = self._extract_lightweight(file_path, logger)
        elif metrics.recommended_strategy == ProcessingStrategy.STANDARD:
            content, metadata = self._extract_standard(file_path, logger)
        elif metrics.recommended_strategy == ProcessingStrategy.PARALLEL:
            content, metadata = self._extract_parallel(file_path, logger)
        else:  # STREAMING
            content, metadata = self._extract_streaming(file_path, logger)

        processing_time = time.time() - start_time
        metadata["processing_time"] = processing_time
        metadata["strategy_used"] = metrics.recommended_strategy.value

        completion_info = f"Completed in {processing_time:.2f}s (estimated {metrics.estimated_processing_time:.1f}s)"

        # Skip individual file completion for cleaner aggregate display
        if not logger:
            print(completion_info)

        # Cache result
        self.cache.set(file_path, content, metadata)

        return content, metadata

    def _extract_lightweight(self, file_path: Path, logger: Optional["HierarchicalLogger"] = None) -> Tuple[str, Dict[str, Any]]:
        """Lightweight processing for small PDFs.

        Args:
            file_path: Path to the PDF file
            logger: Optional HierarchicalLogger for progress display

        Returns:
            Tuple of (content, metadata)

        """
        doc = pymupdf.open(file_path)
        self._handle_encryption(doc, file_path)

        # Use pymupdf4llm for direct markdown conversion
        md_text = pymupdf4llm.to_markdown(doc)

        metadata = self._extract_metadata(doc)
        doc.close()

        return md_text, metadata

    def _extract_standard(self, file_path: Path, logger: Optional["HierarchicalLogger"] = None) -> Tuple[str, Dict[str, Any]]:
        """Process PDF using standard strategy with basic optimizations.

        Args:
            file_path: Path to the PDF file
            logger: Optional HierarchicalLogger for progress display

        Returns:
            Tuple of (content, metadata)

        """
        return self._extract_lightweight(file_path, logger)  # Same as lightweight for now

    def _extract_parallel(self, file_path: Path, logger: Optional["HierarchicalLogger"] = None) -> Tuple[str, Dict[str, Any]]:
        """Parallel processing for medium-large PDFs.

        Args:
            file_path: Path to the PDF file
            logger: Optional HierarchicalLogger for progress display

        Returns:
            Tuple of (content, metadata)

        """
        doc = pymupdf.open(file_path)
        self._handle_encryption(doc, file_path)

        total_pages = doc.page_count
        # Skip individual PDF operation for cleaner aggregate display

        # Use page_chunks option for parallel processing
        page_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)

        # Parallel processing of page chunks
        markdown_content: List[Optional[str]] = [None] * total_pages

        def process_page_chunk(chunk_data: Tuple[int, Dict[str, Any]]) -> Tuple[int, str]:
            page_num, chunk = chunk_data
            try:
                # Each chunk contains 'text' and metadata
                text = chunk.get("text", "")
                return page_num, text if text and text.strip() else ""
            except Exception as e:
                print(f"Warning: Failed to process page {page_num + 1}: {e}")
                return page_num, ""

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            chunk_data = [(i, chunk) for i, chunk in enumerate(page_chunks)]

            # Submit all tasks
            future_to_page = {executor.submit(process_page_chunk, data): data[0] for data in chunk_data}

            # Collect results with progress
            completed = 0
            for future in as_completed(future_to_page):
                page_num, text = future.result()
                markdown_content[page_num] = text

                completed += 1
                progress = (completed / total_pages) * 100

                # Skip individual progress updates for cleaner aggregate display
                if completed % max(1, total_pages // 20) == 0 or completed == total_pages:
                    if not logger:
                        progress_info = f"Processing pages ({completed}/{total_pages}, {progress:.0f}%)"
                        print(f"  {progress_info}")

        # Filter out None values and join
        valid_content = [text for text in markdown_content if text is not None and text]
        content = "\n\n---\n\n".join(valid_content)  # Use markdown page separator

        metadata = self._extract_metadata(doc)
        metadata["extracted_pages"] = len(valid_content)
        doc.close()

        # Skip individual operation completion for cleaner aggregate display
        return content, metadata

    def _extract_streaming(self, file_path: Path, logger: Optional["HierarchicalLogger"] = None) -> Tuple[str, Dict[str, Any]]:
        """Memory-efficient streaming processing for large PDFs.

        Args:
            file_path: Path to the PDF file
            logger: Optional HierarchicalLogger for progress display

        Returns:
            Tuple of (content, metadata)

        """
        # For very large PDFs, process in chunks to limit memory usage
        chunk_size = 20  # Process 20 pages at a time

        doc = pymupdf.open(file_path)
        self._handle_encryption(doc, file_path)

        total_pages = doc.page_count
        # Skip individual PDF operation for cleaner aggregate display

        all_content = []
        processed_pages = 0

        # Process in chunks
        for chunk_start in range(0, total_pages, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pages)

            # Extract pages for this chunk
            pages_list = list(range(chunk_start, chunk_end))

            # Process chunk with pymupdf4llm
            chunk_md = pymupdf4llm.to_markdown(doc, pages=pages_list, page_chunks=True)

            # Process chunk in parallel
            def extract_chunk_page(page_data: Tuple[int, Dict[str, Any]]) -> str:
                page_num, chunk = page_data
                try:
                    return chunk.get("text", "") or ""
                except Exception:
                    return ""

            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunk_md))) as executor:
                chunk_data = [(i, chunk) for i, chunk in enumerate(chunk_md)]
                chunk_results = list(executor.map(extract_chunk_page, chunk_data))

                chunk_content = [text for text in chunk_results if text.strip()]

            all_content.extend(chunk_content)
            processed_pages += len(pages_list)

            progress = (processed_pages / total_pages) * 100
            if not logger:
                progress_info = f"Processing chunks ({processed_pages}/{total_pages}, {progress:.0f}%)"
                print(f"  {progress_info}")

        content = "\n\n---\n\n".join(all_content)  # Use markdown page separator

        metadata = self._extract_metadata(doc)
        metadata["extracted_pages"] = len(all_content)
        doc.close()

        # Skip individual operation completion for cleaner aggregate display
        return content, metadata

    def _handle_encryption(self, doc: "pymupdf.Document", file_path: Path) -> None:
        """Handle PDF encryption."""
        if doc.is_encrypted:
            if not doc.authenticate(""):
                raise RuntimeError(f"PDF file is password-protected: {file_path.name}")

    def _extract_metadata(self, doc: "pymupdf.Document") -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {"total_pages": doc.page_count}

        # Extract document metadata
        doc_metadata = doc.metadata
        if doc_metadata:
            try:
                if doc_metadata.get("title"):
                    metadata["title"] = doc_metadata["title"]
                if doc_metadata.get("author"):
                    metadata["creator"] = doc_metadata["author"]
                if doc_metadata.get("creationDate"):
                    # PyMuPDF returns dates as strings
                    metadata["creation_date"] = doc_metadata["creationDate"]
                if doc_metadata.get("modDate"):
                    metadata["modification_date"] = doc_metadata["modDate"]
            except Exception:  # noqa: S110
                pass

        return metadata

    async def extract_pdf_async(self, file_path: Path, logger: Optional["HierarchicalLogger"] = None) -> Tuple[str, Dict[str, Any]]:
        """Async wrapper for PDF extraction.

        Args:
            file_path: Path to the PDF file
            logger: Optional HierarchicalLogger for progress display

        Returns:
            Tuple of (content, metadata)

        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_pdf, file_path, logger)

    def batch_process_pdfs(self, pdf_paths: List[Path], max_concurrent: int = 2) -> Iterator[Tuple[Path, Tuple[str, Dict[str, Any]]]]:
        """Batch process multiple PDFs with concurrency control."""
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_path = {executor.submit(self.extract_pdf, path): path for path in pdf_paths}

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    yield path, result
                except Exception:  # noqa: S112
                    # Silently skip problematic files for cleaner display
                    continue
