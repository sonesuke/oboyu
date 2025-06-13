"""Enhanced PDF processor with performance optimizations."""

import asyncio
import hashlib
import os
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
    """Enhanced PDF processing strategies."""
    
    LIGHTWEIGHT = "lightweight"  # <5MB, <50 pages
    STANDARD = "standard"        # 5-15MB, 50-100 pages
    PARALLEL = "parallel"        # 15-30MB, 100-200 pages
    STREAMING = "streaming"      # >30MB, >200 pages


@dataclass
class PDFMetrics:
    """Enhanced PDF file metrics."""
    
    file_size_mb: float
    total_pages: int
    estimated_processing_time: float
    recommended_strategy: ProcessingStrategy
    optimal_workers: int
    
    @classmethod
    def analyze_pdf(cls, file_path: Path) -> "PDFMetrics":
        """Analyze PDF with enhanced metrics calculation."""
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        # Quick page count estimation
        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                total_pages = len(reader.pages)
        except Exception:
            total_pages = max(1, int(file_size_mb * 10))
        
        # Enhanced time estimation based on analysis
        estimated_time = (file_size_mb * 0.3) + (total_pages * 0.01)  # More accurate
        
        # Enhanced strategy selection
        if file_size_mb < 2 or total_pages < 20:
            strategy = ProcessingStrategy.LIGHTWEIGHT
            optimal_workers = 1
        elif file_size_mb < 10 or total_pages < 75:
            strategy = ProcessingStrategy.STANDARD
            optimal_workers = min(2, os.cpu_count() or 1)
        elif file_size_mb < 25 or total_pages < 150:
            strategy = ProcessingStrategy.PARALLEL
            optimal_workers = min(max(total_pages // 25, 2), os.cpu_count() or 4)
        else:
            strategy = ProcessingStrategy.STREAMING
            optimal_workers = min(4, os.cpu_count() or 4)
        
        return cls(file_size_mb, total_pages, estimated_time, strategy, optimal_workers)


class FastTextExtractor:
    """Optimized text extraction for PDF pages."""
    
    @staticmethod
    def extract_text_fast(page: Any) -> str:
        """Extract text fast with error handling."""
        try:
            # Try standard extraction first
            text = page.extract_text()
            if text and text.strip():
                return text.strip()
            
            # Fallback to simple text extraction if standard fails
            return FastTextExtractor._extract_text_simple(page)
        except Exception:
            return ""
    
    @staticmethod
    def _extract_text_simple(page: Any) -> str:
        """Extract text using simple fallback method."""
        try:
            # Get the page's content stream
            if "/Contents" in page:
                content = page["/Contents"]
                if hasattr(content, 'get_data'):
                    # Try to extract text from raw content
                    raw_data = content.get_data()
                    # This is a simplified approach - in production we'd parse PDF operators
                    return str(raw_data)[:1000]  # Limit to prevent memory issues
        except Exception:
            pass
        return ""
    
    @staticmethod
    def batch_extract_pages(pages: List[Any], max_workers: int = 4) -> List[str]:
        """Extract text from multiple pages efficiently."""
        if len(pages) <= 2:
            # For small batches, process sequentially to avoid overhead
            return [FastTextExtractor.extract_text_fast(page) for page in pages]
        
        # Parallel processing for larger batches
        results: List[Optional[str]] = [None] * len(pages)
        
        def extract_page_with_index(page_data: Tuple[int, Any]) -> Tuple[int, str]:
            index, page = page_data
            return index, FastTextExtractor.extract_text_fast(page)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            page_data = [(i, page) for i, page in enumerate(pages)]
            
            # Use map for better performance than submit/as_completed for this use case
            for index, text in executor.map(extract_page_with_index, page_data):
                results[index] = text
        
        return [text or "" for text in results]


class EnhancedPDFCache:
    """Enhanced caching with better performance."""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_cache_size_mb: int = 500) -> None:
        """Initialize enhanced cache."""
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "oboyu_pdf_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size_mb = max_cache_size_mb
    
    def _get_cache_key(self, file_path: Path) -> str:
        """Optimized cache key generation."""
        stat = file_path.stat()
        
        # For files > 10MB, use quick key to avoid expensive hashing
        if stat.st_size > 10 * 1024 * 1024:
            quick_key = f"{file_path.name}_{stat.st_size}_{stat.st_mtime_ns}"
            return hashlib.md5(quick_key.encode()).hexdigest()
        
        # For smaller files, use content hash for accuracy
        try:
            with open(file_path, "rb") as f:
                # Only hash first and last 64KB for large files
                start_chunk = f.read(65536)
                f.seek(-65536, 2) if stat.st_size > 65536 else f.seek(0)
                end_chunk = f.read(65536)
                
            content_hash = hashlib.md5(start_chunk + end_chunk).hexdigest()
            return f"{content_hash}_{stat.st_size}"
        except Exception:
            # Fallback to simple key
            return hashlib.md5(f"{file_path.name}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()
    
    def get(self, file_path: Path) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get cached result with size checking."""
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def set(self, file_path: Path, content: str, metadata: Dict[str, Any]) -> None:
        """Cache result with size management."""
        # Check cache size and clean if needed
        self._manage_cache_size()
        
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((content, metadata), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    
    def _manage_cache_size(self) -> None:
        """Manage cache size by removing old entries."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            if total_size > self.max_cache_size_mb * 1024 * 1024:
                # Sort by modification time and remove oldest
                cache_files.sort(key=lambda f: f.stat().st_mtime)
                
                for cache_file in cache_files[:len(cache_files) // 3]:  # Remove oldest 1/3
                    cache_file.unlink(missing_ok=True)
        except Exception:
            pass


class EnhancedPDFProcessor:
    """Enhanced PDF processor with performance optimizations."""
    
    def __init__(self, max_file_size: int = 100 * 1024 * 1024, cache_size_mb: int = 500) -> None:
        """Initialize enhanced processor."""
        self.max_file_size = max_file_size
        self.cache = EnhancedPDFCache(max_cache_size_mb=cache_size_mb)
        self.text_extractor = FastTextExtractor()
    
    def extract_pdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Enhanced PDF extraction with optimizations."""
        if pypdf is None:
            raise RuntimeError("pypdf library is required for PDF processing")
        
        # Check cache first
        cached_result = self.cache.get(file_path)
        if cached_result:
            print(f"✅ Using cached result for {file_path.name}")
            return cached_result
        
        # Analyze file characteristics
        metrics = PDFMetrics.analyze_pdf(file_path)
        print(
            f"📊 Processing {file_path.name}: {metrics.recommended_strategy.value} strategy "
            f"({metrics.file_size_mb:.1f}MB, {metrics.total_pages} pages, "
            f"{metrics.optimal_workers} workers, ~{metrics.estimated_processing_time:.1f}s estimated)"
        )
        
        # Check file size limit
        if file_path.stat().st_size > self.max_file_size:
            raise RuntimeError(
                f"PDF file too large ({metrics.file_size_mb:.1f}MB). "
                f"Maximum supported size: {self.max_file_size / (1024 * 1024):.1f}MB"
            )
        
        # Select processing strategy
        start_time = time.time()
        
        try:
            if metrics.recommended_strategy == ProcessingStrategy.LIGHTWEIGHT:
                content, metadata = self._extract_lightweight(file_path, metrics)
            elif metrics.recommended_strategy == ProcessingStrategy.STANDARD:
                content, metadata = self._extract_standard(file_path, metrics)
            elif metrics.recommended_strategy == ProcessingStrategy.PARALLEL:
                content, metadata = self._extract_parallel(file_path, metrics)
            else:  # STREAMING
                content, metadata = self._extract_streaming(file_path, metrics)
        except Exception as e:
            print(f"⚠️  Processing error: {e}")
            # Fallback to simple extraction
            content, metadata = self._extract_fallback(file_path)
        
        processing_time = time.time() - start_time
        metadata["processing_time"] = processing_time
        metadata["strategy_used"] = metrics.recommended_strategy.value
        metadata["workers_used"] = metrics.optimal_workers
        
        print(f"✅ Completed in {processing_time:.2f}s (estimated {metrics.estimated_processing_time:.1f}s)")
        
        # Cache result
        self.cache.set(file_path, content, metadata)
        
        return content, metadata
    
    def _extract_lightweight(self, file_path: Path, metrics: PDFMetrics) -> Tuple[str, Dict[str, Any]]:
        """Lightweight processing with fast extraction."""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            self._handle_encryption(reader, file_path)
            
            # Use fast text extractor
            text_content = []
            for page in reader.pages:
                page_text = self.text_extractor.extract_text_fast(page)
                if page_text:
                    text_content.append(page_text)
            
            content = "\n\n".join(text_content)
            metadata = self._extract_metadata(reader, len(text_content))
            
            return content, metadata
    
    def _extract_standard(self, file_path: Path, metrics: PDFMetrics) -> Tuple[str, Dict[str, Any]]:
        """Process PDF using standard strategy with batch extraction."""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            self._handle_encryption(reader, file_path)
            
            # Batch process pages for better efficiency
            pages = list(reader.pages)
            batch_size = min(10, len(pages))
            
            all_text: List[str] = []
            for i in range(0, len(pages), batch_size):
                batch = pages[i:i + batch_size]
                batch_text = self.text_extractor.batch_extract_pages(batch, metrics.optimal_workers)
                all_text.extend(text for text in batch_text if text)
            
            content = "\n\n".join(all_text)
            metadata = self._extract_metadata(reader, len(all_text))
            
            return content, metadata
    
    def _extract_parallel(self, file_path: Path, metrics: PDFMetrics) -> Tuple[str, Dict[str, Any]]:
        """Enhanced parallel processing."""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            self._handle_encryption(reader, file_path)
            
            pages = list(reader.pages)
            total_pages = len(pages)
            
            print(f"🔧 Processing {total_pages} pages with {metrics.optimal_workers} workers")
            
            # Use batch processing for better efficiency
            text_content = self.text_extractor.batch_extract_pages(pages, metrics.optimal_workers)
            
            # Filter out empty content
            valid_content = [text for text in text_content if text]
            content = "\n\n".join(valid_content)
            metadata = self._extract_metadata(reader, len(valid_content))
            
            return content, metadata
    
    def _extract_streaming(self, file_path: Path, metrics: PDFMetrics) -> Tuple[str, Dict[str, Any]]:
        """Enhanced streaming processing."""
        chunk_size = max(20, metrics.total_pages // 10)  # Adaptive chunk size
        
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            self._handle_encryption(reader, file_path)
            
            total_pages = len(reader.pages)
            print(f"🌊 Streaming processing {total_pages} pages in chunks of {chunk_size}")
            
            all_content = []
            processed_pages = 0
            
            for chunk_start in range(0, total_pages, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_pages)
                chunk_pages = reader.pages[chunk_start:chunk_end]
                
                # Process chunk with optimal workers
                chunk_text = self.text_extractor.batch_extract_pages(
                    list(chunk_pages),
                    min(metrics.optimal_workers, len(chunk_pages))
                )
                
                valid_chunk_text = [text for text in chunk_text if text]
                all_content.extend(valid_chunk_text)
                processed_pages += len(chunk_pages)
                
                progress = (processed_pages / total_pages) * 100
                print(f"   📊 Chunk progress: {progress:.0f}% ({processed_pages}/{total_pages} pages)")
            
            content = "\n\n".join(all_content)
            metadata = self._extract_metadata(reader, len(all_content))
            
            return content, metadata
    
    def _extract_fallback(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Fallback extraction method for problematic PDFs."""
        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                
                # Simple extraction without error handling
                text_parts = []
                pages_processed = 0
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_parts.append(text.strip())
                            pages_processed += 1
                    except Exception:
                        # Skip problematic pages
                        continue
                
                content = "\n\n".join(text_parts)
                metadata = {
                    "total_pages": len(reader.pages),
                    "extracted_pages": pages_processed,
                    "fallback_extraction": True
                }
                
                return content, metadata
        except Exception:
            return "", {"error": "Fallback extraction failed", "total_pages": 0, "extracted_pages": 0}
    
    def _handle_encryption(self, reader: "pypdf.PdfReader", file_path: Path) -> None:
        """Handle PDF encryption."""
        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise RuntimeError(f"PDF file is password-protected: {file_path.name}")
    
    def _extract_metadata(self, reader: "pypdf.PdfReader", extracted_pages: int) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {"total_pages": len(reader.pages), "extracted_pages": extracted_pages}
        
        if reader.metadata:
            try:
                for key, attr in [
                    ("title", "title"),
                    ("creator", "creator"),
                    ("creation_date", "creation_date"),
                    ("modification_date", "modification_date"),
                ]:
                    value = getattr(reader.metadata, attr, None)
                    if value:
                        if hasattr(value, 'isoformat'):
                            metadata[key] = value.isoformat()
                        else:
                            metadata[key] = str(value)
            except Exception:
                pass
        
        return metadata
    
    async def extract_pdf_async(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Async wrapper for PDF extraction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_pdf, file_path)
    
    def batch_process_pdfs(self, pdf_paths: List[Path], max_concurrent: int = 3) -> Iterator[Tuple[Path, Tuple[str, Dict[str, Any]]]]:
        """Enhanced batch processing."""
        # Adaptive concurrency based on system resources
        optimal_concurrent = min(max_concurrent, max(1, (os.cpu_count() or 2) // 2))
        
        with ThreadPoolExecutor(max_workers=optimal_concurrent) as executor:
            future_to_path = {executor.submit(self.extract_pdf, path): path for path in pdf_paths}
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    yield path, result
                except Exception as e:
                    print(f"❌ Error processing {path}: {e}")
                    continue
