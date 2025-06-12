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
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import pypdf
except ImportError:
    pypdf = None


class ProcessingStrategy(Enum):
    """PDF processing strategies based on file characteristics."""

    LIGHTWEIGHT = "lightweight"  # <5MB, <50 pages
    STANDARD = "standard"        # 5-15MB, 50-100 pages
    PARALLEL = "parallel"        # 15-30MB, 100-200 pages
    STREAMING = "streaming"      # >30MB, >200 pages


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
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                total_pages = len(reader.pages)
        except Exception:
            # Fallback estimation based on file size
            total_pages = max(1, int(file_size_mb * 10))  # Rough estimate
        
        # Estimate processing time based on test results
        estimated_time = (file_size_mb * 0.57) + (total_pages * 0.086)
        
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
        
    def extract_pdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract content and metadata from PDF file."""
        if pypdf is None:
            raise RuntimeError("pypdf library is required for PDF processing")
        
        # Check cache first
        cached_result = self.cache.get(file_path)
        if cached_result:
            print(f"Using cached result for {file_path.name}")
            return cached_result
        
        # Analyze file characteristics
        metrics = PDFMetrics.analyze_pdf(file_path)
        print(f"Processing {file_path.name}: {metrics.recommended_strategy.value} strategy "
              f"({metrics.file_size_mb:.1f}MB, {metrics.total_pages} pages, "
              f"~{metrics.estimated_processing_time:.1f}s estimated)")
        
        # Check file size limit
        if file_path.stat().st_size > self.max_file_size:
            raise RuntimeError(f"PDF file too large ({metrics.file_size_mb:.1f}MB). "
                             f"Maximum supported size: {self.max_file_size / (1024*1024):.1f}MB")
        
        # Select processing strategy
        start_time = time.time()
        
        if metrics.recommended_strategy == ProcessingStrategy.LIGHTWEIGHT:
            content, metadata = self._extract_lightweight(file_path)
        elif metrics.recommended_strategy == ProcessingStrategy.STANDARD:
            content, metadata = self._extract_standard(file_path)
        elif metrics.recommended_strategy == ProcessingStrategy.PARALLEL:
            content, metadata = self._extract_parallel(file_path)
        else:  # STREAMING
            content, metadata = self._extract_streaming(file_path)
        
        processing_time = time.time() - start_time
        metadata["processing_time"] = processing_time
        metadata["strategy_used"] = metrics.recommended_strategy.value
        
        print(f"Completed in {processing_time:.2f}s (estimated {metrics.estimated_processing_time:.1f}s)")
        
        # Cache result
        self.cache.set(file_path, content, metadata)
        
        return content, metadata
    
    def _extract_lightweight(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Lightweight processing for small PDFs."""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            self._handle_encryption(reader, file_path)
            
            text_content = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_content.append(page_text)
                except Exception:  # noqa: S112
                    continue
            
            content = "\n\n".join(text_content)
            metadata = self._extract_metadata(reader, len(text_content))
            
            return content, metadata
    
    def _extract_standard(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Process PDF using standard strategy with basic optimizations."""
        return self._extract_lightweight(file_path)  # Same as lightweight for now
    
    def _extract_parallel(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parallel processing for medium-large PDFs."""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            self._handle_encryption(reader, file_path)
            
            pages = reader.pages
            total_pages = len(pages)
            
            print(f"Processing {total_pages} pages in parallel with {self.max_workers} workers")
            
            # Parallel page processing
            text_content: List[Optional[str]] = [None] * total_pages
            
            def extract_page_text(page_data: Tuple[int, Any]) -> Tuple[int, str]:
                page_num, page = page_data
                try:
                    text = page.extract_text()
                    return page_num, text if text and text.strip() else ""
                except Exception as e:
                    print(f"Warning: Failed to extract page {page_num + 1}: {e}")
                    return page_num, ""
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                page_data = [(i, page) for i, page in enumerate(pages)]
                
                # Submit all tasks
                future_to_page = {
                    executor.submit(extract_page_text, data): data[0]
                    for data in page_data
                }
                
                # Collect results with progress
                completed = 0
                for future in as_completed(future_to_page):
                    page_num, text = future.result()
                    text_content[page_num] = text
                    
                    completed += 1
                    if completed % max(1, total_pages // 10) == 0:
                        progress = (completed / total_pages) * 100
                        print(f"  Progress: {progress:.0f}% ({completed}/{total_pages} pages)")
            
            # Filter out None values and join
            valid_content = [text for text in text_content if text is not None and text]
            content = "\n\n".join(valid_content)
            metadata = self._extract_metadata(reader, len(valid_content))
            
            return content, metadata
    
    def _extract_streaming(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Memory-efficient streaming processing for large PDFs."""
        # For very large PDFs, process in chunks to limit memory usage
        chunk_size = 20  # Process 20 pages at a time
        
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            self._handle_encryption(reader, file_path)
            
            total_pages = len(reader.pages)
            print(f"Streaming processing {total_pages} pages in chunks of {chunk_size}")
            
            all_content = []
            processed_pages = 0
            
            # Process in chunks
            for chunk_start in range(0, total_pages, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_pages)
                chunk_pages = reader.pages[chunk_start:chunk_end]
                
                # Process chunk in parallel
                chunk_content = []
                
                def extract_chunk_page(page_data: Tuple[int, Any]) -> str:
                    page_num, page = page_data
                    try:
                        return page.extract_text() or ""
                    except Exception:
                        return ""
                
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunk_pages))) as executor:
                    chunk_data = [(i, page) for i, page in enumerate(chunk_pages)]
                    chunk_results = list(executor.map(extract_chunk_page, chunk_data))
                    
                    chunk_content = [text for text in chunk_results if text.strip()]
                
                all_content.extend(chunk_content)
                processed_pages += len(chunk_pages)
                
                progress = (processed_pages / total_pages) * 100
                print(f"  Chunk progress: {progress:.0f}% ({processed_pages}/{total_pages} pages)")
            
            content = "\n\n".join(all_content)
            metadata = self._extract_metadata(reader, len(all_content))
            
            return content, metadata
    
    def _handle_encryption(self, reader: "pypdf.PdfReader", file_path: Path) -> None:
        """Handle PDF encryption."""
        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise RuntimeError(f"PDF file is password-protected: {file_path.name}")
    
    def _extract_metadata(self, reader: "pypdf.PdfReader", extracted_pages: int) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {
            "total_pages": len(reader.pages),
            "extracted_pages": extracted_pages
        }
        
        if reader.metadata:
            try:
                if reader.metadata.title:
                    metadata["title"] = reader.metadata.title
                if reader.metadata.creator:
                    metadata["creator"] = reader.metadata.creator
                if reader.metadata.creation_date:
                    metadata["creation_date"] = reader.metadata.creation_date.isoformat()
                if reader.metadata.modification_date:
                    metadata["modification_date"] = reader.metadata.modification_date.isoformat()
            except Exception:  # noqa: S110
                pass
        
        return metadata
    
    async def extract_pdf_async(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Async wrapper for PDF extraction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_pdf, file_path)
    
    def batch_process_pdfs(self, pdf_paths: List[Path], max_concurrent: int = 2) -> Iterator[Tuple[Path, Tuple[str, Dict[str, Any]]]]:
        """Batch process multiple PDFs with concurrency control."""
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_path = {
                executor.submit(self.extract_pdf, path): path
                for path in pdf_paths
            }
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    yield path, result
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    continue
