#!/usr/bin/env python3
"""PDF performance analysis with reliable test PDFs."""

import cProfile
import io
import pstats
import time
from pathlib import Path

from oboyu.crawler.services.optimized_pdf_processor import OptimizedPDFProcessor


def analyze_reliable_pdf_performance():
    """Analyze PDF processing performance with reliable test files."""
    print("🔍 PDF Performance Analysis (Fixed Test Files)")
    print("=" * 60)
    
    processor = OptimizedPDFProcessor()
    
    # Use the reliable test PDFs
    test_dir = Path("tests/fixtures/pdf/performance_tests")
    test_files = [
        ("small_reliable.pdf", "Small (5 pages)"),
        ("medium_reliable.pdf", "Medium (25 pages)"),
        ("large_reliable.pdf", "Large (100 pages)"),
        ("xl_reliable.pdf", "XL (200 pages)"),
    ]
    
    results = []
    
    for filename, description in test_files:
        pdf_path = test_dir / filename
        
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            continue
        
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"\n📄 Testing {description}")
        print(f"   File size: {file_size_mb:.3f} MB")
        
        # Profile the processing
        profiler = cProfile.Profile()
        
        profiler.enable()
        start_time = time.time()
        
        try:
            content, metadata = processor.extract_pdf(pdf_path)
            success = True
            error = None
        except Exception as e:
            content, metadata = "", {}
            success = False
            error = str(e)
        
        end_time = time.time()
        profiler.disable()
        
        processing_time = end_time - start_time
        throughput = file_size_mb / processing_time if processing_time > 0 else 0
        
        # Get profiling stats
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        result = {
            'name': description,
            'filename': filename,
            'file_size_mb': file_size_mb,
            'processing_time': processing_time,
            'throughput_mb_per_sec': throughput,
            'content_length': len(content),
            'pages_processed': metadata.get('extracted_pages', 0),
            'total_pages': metadata.get('total_pages', 0),
            'strategy': metadata.get('strategy_used', 'unknown'),
            'success': success,
            'error': error,
            'profile_stats': stats_buffer.getvalue()
        }
        
        results.append(result)
        
        print(f"   ⏱️  Processing time: {processing_time:.3f}s")
        print(f"   🚀 Throughput: {throughput:.2f} MB/s")
        print(f"   📊 Strategy: {result['strategy']}")
        print(f"   📝 Content: {len(content):,} characters")
        print(f"   ✅ Pages: {result['pages_processed']}/{result['total_pages']}")
        
        if not success:
            print(f"   ❌ Error: {error}")
        elif result['pages_processed'] < result['total_pages']:
            failed_pages = result['total_pages'] - result['pages_processed']
            print(f"   ⚠️  Failed pages: {failed_pages}")
    
    # Performance summary
    print("\n" + "="*60)
    print("📈 PERFORMANCE SUMMARY")
    print("="*60)
    
    if results:
        print(f"{'Test':<20} {'Size(MB)':<10} {'Time(s)':<10} {'Throughput':<12} {'Strategy':<12} {'Success':<8}")
        print("-" * 78)
        
        for result in results:
            status = "✅ OK" if result['success'] and result['pages_processed'] == result['total_pages'] else "⚠️ Issues"
            print(f"{result['name']:<20} {result['file_size_mb']:<10.3f} {result['processing_time']:<10.3f} "
                  f"{result['throughput_mb_per_sec']:<12.2f} {result['strategy']:<12} {status:<8}")
    
    # Performance trends
    print("\n🎯 PERFORMANCE TRENDS:")
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) >= 2:
        throughputs = [r['throughput_mb_per_sec'] for r in successful_results]
        times_per_page = [r['processing_time'] / r['total_pages'] for r in successful_results]
        
        print(f"   Throughput range: {min(throughputs):.2f} - {max(throughputs):.2f} MB/s")
        print(f"   Time per page range: {min(times_per_page):.4f} - {max(times_per_page):.4f}s")
        
        # Check for performance scaling
        if throughputs[-1] >= throughputs[0] * 0.8:  # Within 20%
            print("   ✅ Throughput scales well with file size")
        else:
            print("   ⚠️  Throughput degrades with larger files")
    
    # Key optimizations identified
    print("\n💡 KEY FINDINGS:")
    
    # Check for page extraction failures
    total_pages = sum(r['total_pages'] for r in results if r['success'])
    failed_pages = sum(r['total_pages'] - r['pages_processed'] for r in results if r['success'])
    
    if failed_pages == 0:
        print("   ✅ All pages extracted successfully (PDF generation fixed)")
    else:
        print(f"   ⚠️  {failed_pages}/{total_pages} pages failed extraction")
    
    # Analyze processing times
    if successful_results:
        avg_time_per_page = sum(r['processing_time'] / r['total_pages'] for r in successful_results) / len(successful_results)
        if avg_time_per_page < 0.01:  # < 10ms per page
            print(f"   ✅ Excellent processing speed: {avg_time_per_page:.4f}s per page")
        elif avg_time_per_page < 0.05:  # < 50ms per page
            print(f"   ✅ Good processing speed: {avg_time_per_page:.4f}s per page")
        else:
            print(f"   ⚠️  Slow processing: {avg_time_per_page:.4f}s per page")
    
    return results


if __name__ == "__main__":
    results = analyze_reliable_pdf_performance()
    
    print("\n🏁 Analysis complete!")
    print(f"   Processed {len(results)} test files")
    print("   Results saved for further optimization")
