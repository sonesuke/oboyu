"""PDF performance testing and benchmarking."""

import cProfile
import io
import pstats
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from oboyu.crawler.services.optimized_pdf_processor import OptimizedPDFProcessor, ProcessingStrategy
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.pdf.create_test_pdfs import create_large_pdf, create_multipage_pdf


class PDFPerformanceTester:
    """Comprehensive PDF performance testing utility."""
    
    def __init__(self) -> None:
        """Initialize performance tester."""
        self.processor = OptimizedPDFProcessor()
        self.results: List[Dict] = []
    
    def profile_pdf_processing(self, pdf_path: Path) -> Dict:
        """Profile PDF processing with detailed metrics."""
        profiler = cProfile.Profile()
        
        # Start profiling
        profiler.enable()
        start_time = time.time()
        
        try:
            content, metadata = self.processor.extract_pdf(pdf_path)
            success = True
            error = None
        except Exception as e:
            content, metadata = "", {}
            success = False
            error = str(e)
        
        end_time = time.time()
        profiler.disable()
        
        # Analyze profiling results
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        profile_output = stats_buffer.getvalue()
        
        # Calculate metrics
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        processing_time = end_time - start_time
        throughput_mb_per_sec = file_size_mb / processing_time if processing_time > 0 else 0
        
        result = {
            'file_path': str(pdf_path),
            'file_size_mb': file_size_mb,
            'processing_time': processing_time,
            'throughput_mb_per_sec': throughput_mb_per_sec,
            'content_length': len(content),
            'pages_processed': metadata.get('extracted_pages', 0),
            'total_pages': metadata.get('total_pages', 0),
            'strategy_used': metadata.get('strategy_used', 'unknown'),
            'success': success,
            'error': error,
            'profile_output': profile_output
        }
        
        self.results.append(result)
        return result
    
    def benchmark_processing_strategies(self, test_pdfs: List[Path]) -> Dict:
        """Benchmark different processing strategies."""
        strategy_results = {}
        
        for pdf_path in test_pdfs:
            result = self.profile_pdf_processing(pdf_path)
            strategy = result['strategy_used']
            
            if strategy not in strategy_results:
                strategy_results[strategy] = []
            
            strategy_results[strategy].append(result)
        
        # Calculate strategy averages
        strategy_summary = {}
        for strategy, results in strategy_results.items():
            if results:
                avg_throughput = sum(r['throughput_mb_per_sec'] for r in results) / len(results)
                avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
                avg_file_size = sum(r['file_size_mb'] for r in results) / len(results)
                
                strategy_summary[strategy] = {
                    'count': len(results),
                    'avg_throughput_mb_per_sec': avg_throughput,
                    'avg_processing_time': avg_processing_time,
                    'avg_file_size_mb': avg_file_size,
                    'results': results
                }
        
        return strategy_summary
    
    def memory_usage_test(self, pdf_path: Path) -> Dict:
        """Test memory usage during PDF processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Process PDF
        start_time = time.time()
        content, metadata = self.processor.extract_pdf(pdf_path)
        end_time = time.time()
        
        # Peak memory (approximate)
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = peak_memory - baseline_memory
        
        return {
            'file_path': str(pdf_path),
            'file_size_mb': pdf_path.stat().st_size / (1024 * 1024),
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': memory_increase,
            'processing_time': end_time - start_time,
            'content_length': len(content),
            'pages_processed': metadata.get('extracted_pages', 0)
        }
    
    def concurrent_processing_test(self, pdf_paths: List[Path], max_concurrent: int = 3) -> Dict:
        """Test concurrent PDF processing performance."""
        start_time = time.time()
        
        results = list(self.processor.batch_process_pdfs(pdf_paths, max_concurrent))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        total_files = len(pdf_paths)
        successful_files = len(results)
        total_size_mb = sum(p.stat().st_size for p in pdf_paths) / (1024 * 1024)
        
        return {
            'total_files': total_files,
            'successful_files': successful_files,
            'total_size_mb': total_size_mb,
            'total_processing_time': total_time,
            'throughput_files_per_sec': successful_files / total_time if total_time > 0 else 0,
            'throughput_mb_per_sec': total_size_mb / total_time if total_time > 0 else 0,
            'max_concurrent': max_concurrent,
            'individual_results': results
        }


@pytest.fixture
def performance_tester():
    """Create performance tester instance."""
    return PDFPerformanceTester()


@pytest.fixture
def test_pdfs(tmp_path):
    """Create various test PDFs for performance testing."""
    pdfs = []
    
    # Small PDF (lightweight strategy)
    small_pdf = tmp_path / "small.pdf"
    create_multipage_pdf(small_pdf, num_pages=5)
    pdfs.append(small_pdf)
    
    # Medium PDF (standard strategy)
    medium_pdf = tmp_path / "medium.pdf"
    create_multipage_pdf(medium_pdf, num_pages=25)
    pdfs.append(medium_pdf)
    
    # Large PDF (parallel strategy)
    large_pdf = tmp_path / "large.pdf"
    create_large_pdf(large_pdf, num_pages=150, content_per_page=1000)
    pdfs.append(large_pdf)
    
    # Very large PDF (streaming strategy)
    xl_pdf = tmp_path / "extra_large.pdf"
    create_large_pdf(xl_pdf, num_pages=250, content_per_page=1500)
    pdfs.append(xl_pdf)
    
    return pdfs


@pytest.mark.slow
def test_pdf_processing_performance(performance_tester, test_pdfs):
    """Test PDF processing performance across different file sizes."""
    print("\n=== PDF Processing Performance Test ===")
    
    for pdf_path in test_pdfs:
        result = performance_tester.profile_pdf_processing(pdf_path)
        
        print(f"\nFile: {result['file_path']}")
        print(f"Size: {result['file_size_mb']:.2f} MB")
        print(f"Strategy: {result['strategy_used']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"Throughput: {result['throughput_mb_per_sec']:.2f} MB/s")
        print(f"Pages: {result['pages_processed']}/{result['total_pages']}")
        print(f"Content Length: {result['content_length']:,} chars")
        
        # Performance assertions
        if result['file_size_mb'] < 1:
            assert result['processing_time'] < 10, f"Small PDF took too long: {result['processing_time']}s"
        elif result['file_size_mb'] < 10:
            assert result['processing_time'] < 30, f"Medium PDF took too long: {result['processing_time']}s"
        
        assert result['success'], f"Processing failed: {result['error']}"
        assert result['throughput_mb_per_sec'] > 0.1, "Throughput too low"


@pytest.mark.slow
def test_processing_strategy_comparison(performance_tester, test_pdfs):
    """Compare performance across different processing strategies."""
    print("\n=== Processing Strategy Comparison ===")
    
    strategy_results = performance_tester.benchmark_processing_strategies(test_pdfs)
    
    for strategy, summary in strategy_results.items():
        print(f"\nStrategy: {strategy}")
        print(f"Files processed: {summary['count']}")
        print(f"Average file size: {summary['avg_file_size_mb']:.2f} MB")
        print(f"Average processing time: {summary['avg_processing_time']:.2f}s")
        print(f"Average throughput: {summary['avg_throughput_mb_per_sec']:.2f} MB/s")
    
    # Assert strategies are working as expected
    assert 'lightweight' in strategy_results or 'standard' in strategy_results
    
    # Verify throughput improves with better strategies for larger files
    if 'lightweight' in strategy_results and 'parallel' in strategy_results:
        lightweight_throughput = strategy_results['lightweight']['avg_throughput_mb_per_sec']
        parallel_throughput = strategy_results['parallel']['avg_throughput_mb_per_sec']
        
        # Parallel should generally be more efficient for larger files
        print(f"Lightweight throughput: {lightweight_throughput:.2f} MB/s")
        print(f"Parallel throughput: {parallel_throughput:.2f} MB/s")


@pytest.mark.slow
def test_memory_usage(performance_tester, test_pdfs):
    """Test memory usage during PDF processing."""
    print("\n=== Memory Usage Test ===")
    
    for pdf_path in test_pdfs:
        result = performance_tester.memory_usage_test(pdf_path)
        
        print(f"\nFile: {Path(result['file_path']).name}")
        print(f"File size: {result['file_size_mb']:.2f} MB")
        print(f"Memory increase: {result['memory_increase_mb']:.2f} MB")
        print(f"Memory efficiency: {result['file_size_mb'] / result['memory_increase_mb']:.2f}x")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
        # Memory usage should be reasonable
        memory_ratio = result['memory_increase_mb'] / result['file_size_mb']
        assert memory_ratio < 10, f"Memory usage too high: {memory_ratio:.2f}x file size"


@pytest.mark.slow
def test_concurrent_processing_performance(performance_tester, test_pdfs):
    """Test concurrent processing performance."""
    print("\n=== Concurrent Processing Test ===")
    
    # Test different concurrency levels
    for max_concurrent in [1, 2, 4]:
        result = performance_tester.concurrent_processing_test(test_pdfs, max_concurrent)
        
        print(f"\nConcurrency level: {max_concurrent}")
        print(f"Total files: {result['total_files']}")
        print(f"Successful files: {result['successful_files']}")
        print(f"Total size: {result['total_size_mb']:.2f} MB")
        print(f"Total time: {result['total_processing_time']:.2f}s")
        print(f"Throughput: {result['throughput_mb_per_sec']:.2f} MB/s")
        print(f"Files per second: {result['throughput_files_per_sec']:.2f}")
        
        assert result['successful_files'] == result['total_files'], "Some files failed to process"
        assert result['throughput_mb_per_sec'] > 0, "No throughput measured"


@pytest.mark.slow
def test_cache_performance(performance_tester, test_pdfs):
    """Test caching performance improvements."""
    print("\n=== Cache Performance Test ===")
    
    if not test_pdfs:
        pytest.skip("No test PDFs available")
    
    test_pdf = test_pdfs[0]  # Use first PDF
    
    # First run (no cache)
    performance_tester.processor.cache.cache_dir.mkdir(exist_ok=True)
    for cache_file in performance_tester.processor.cache.cache_dir.glob("*.pkl"):
        cache_file.unlink()
    
    start_time = time.time()
    content1, metadata1 = performance_tester.processor.extract_pdf(test_pdf)
    first_run_time = time.time() - start_time
    
    # Second run (with cache)
    start_time = time.time()
    content2, metadata2 = performance_tester.processor.extract_pdf(test_pdf)
    second_run_time = time.time() - start_time
    
    print(f"First run (no cache): {first_run_time:.3f}s")
    print(f"Second run (cached): {second_run_time:.3f}s")
    print(f"Cache speedup: {first_run_time / second_run_time:.1f}x")
    
    # Verify cache works
    assert content1 == content2, "Cached content differs from original"
    assert second_run_time < first_run_time, "Cache didn't improve performance"
    assert second_run_time < 0.1, "Cache lookup too slow"  # Should be very fast


if __name__ == "__main__":
    # Run performance tests directly
    tester = PDFPerformanceTester()
    
    # Create test PDFs
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        test_pdfs = []
        
        # Create test PDFs
        small_pdf = tmp_path / "small.pdf"
        create_multipage_pdf(small_pdf, num_pages=5)
        test_pdfs.append(small_pdf)
        
        medium_pdf = tmp_path / "medium.pdf"
        create_multipage_pdf(medium_pdf, num_pages=25)
        test_pdfs.append(medium_pdf)
        
        print("Running PDF performance tests...")
        
        # Run tests
        for pdf_path in test_pdfs:
            result = tester.profile_pdf_processing(pdf_path)
            print(f"Processed {pdf_path.name}: {result['processing_time']:.2f}s, {result['throughput_mb_per_sec']:.2f} MB/s")