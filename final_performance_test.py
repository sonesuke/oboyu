#!/usr/bin/env python3
"""Final performance test with fresh processing (no cache)."""

import shutil
import tempfile
import time
from pathlib import Path

from oboyu.crawler.services.enhanced_pdf_processor import EnhancedPDFProcessor
from oboyu.crawler.services.optimized_pdf_processor import OptimizedPDFProcessor


def run_final_performance_test():
    """Run final performance test with proper cache clearing."""
    print("🎯 Final PDF Performance Test")
    print("=" * 50)
    
    # Clear all caches
    cache_dirs = [
        Path("/tmp/oboyu_pdf_cache"),
        Path.home() / ".cache" / "oboyu_pdf_cache",
        Path(tempfile.gettempdir()) / "oboyu_pdf_cache"
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"🧹 Cleared cache: {cache_dir}")
    
    # Initialize processors with fresh instances
    original_processor = OptimizedPDFProcessor()
    enhanced_processor = EnhancedPDFProcessor()
    
    # Force disable cache for original processor
    original_processor.cache.cache_dir = Path("/tmp/disabled_cache_orig")
    enhanced_processor.cache.cache_dir = Path("/tmp/disabled_cache_enh")
    
    # Test files
    test_dir = Path("tests/fixtures/pdf/performance_tests")
    test_files = [
        ("medium_reliable.pdf", "Medium (25 pages)"),
        ("large_reliable.pdf", "Large (100 pages)"),
    ]
    
    results = []
    
    for filename, description in test_files:
        pdf_path = test_dir / filename
        
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            continue
        
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"\n📄 Testing {description} ({file_size_mb:.3f} MB)")
        print("-" * 40)
        
        # Test original processor (multiple runs for consistency)
        print("🔧 Original Processor:")
        orig_times = []
        for run in range(3):
            try:
                start_time = time.time()
                content_orig, metadata_orig = original_processor.extract_pdf(pdf_path)
                run_time = time.time() - start_time
                orig_times.append(run_time)
                
                if run == 0:  # Only print details for first run
                    throughput_orig = file_size_mb / run_time if run_time > 0 else 0
                    pages_orig = metadata_orig.get('extracted_pages', 0)
                    total_pages_orig = metadata_orig.get('total_pages', 0)
                    
                    print(f"   ⏱️  Time: {run_time:.3f}s")
                    print(f"   🚀 Throughput: {throughput_orig:.2f} MB/s")
                    print(f"   📝 Content: {len(content_orig):,} chars")
                    print(f"   ✅ Pages: {pages_orig}/{total_pages_orig}")
                
            except Exception as e:
                print(f"   ❌ Error in run {run + 1}: {e}")
                orig_times.append(float('inf'))
        
        avg_orig_time = sum(t for t in orig_times if t != float('inf')) / len([t for t in orig_times if t != float('inf')]) if orig_times else float('inf')
        
        # Test enhanced processor (multiple runs for consistency)
        print("\n⚡ Enhanced Processor:")
        enh_times = []
        for run in range(3):
            try:
                start_time = time.time()
                content_enh, metadata_enh = enhanced_processor.extract_pdf(pdf_path)
                run_time = time.time() - start_time
                enh_times.append(run_time)
                
                if run == 0:  # Only print details for first run
                    throughput_enh = file_size_mb / run_time if run_time > 0 else 0
                    pages_enh = metadata_enh.get('extracted_pages', 0)
                    total_pages_enh = metadata_enh.get('total_pages', 0)
                    
                    print(f"   ⏱️  Time: {run_time:.3f}s")
                    print(f"   🚀 Throughput: {throughput_enh:.2f} MB/s")
                    print(f"   📝 Content: {len(content_enh):,} chars")
                    print(f"   ✅ Pages: {pages_enh}/{total_pages_enh}")
                
            except Exception as e:
                print(f"   ❌ Error in run {run + 1}: {e}")
                enh_times.append(float('inf'))
        
        avg_enh_time = sum(t for t in enh_times if t != float('inf')) / len([t for t in enh_times if t != float('inf')]) if enh_times else float('inf')
        
        # Calculate improvement
        if avg_orig_time != float('inf') and avg_enh_time != float('inf'):
            time_improvement = (avg_orig_time - avg_enh_time) / avg_orig_time * 100
            
            print("\n📊 Performance Analysis:")
            print(f"   Original avg time: {avg_orig_time:.3f}s")
            print(f"   Enhanced avg time: {avg_enh_time:.3f}s")
            print(f"   Time improvement: {time_improvement:+.1f}%")
            print(f"   Times (orig): {[f'{t:.3f}' for t in orig_times]}")
            print(f"   Times (enh): {[f'{t:.3f}' for t in enh_times]}")
            
            results.append({
                "name": description,
                "file_size_mb": file_size_mb,
                "orig_time": avg_orig_time,
                "enh_time": avg_enh_time,
                "improvement": time_improvement
            })
        
        # Clear any created cache files
        for cache_dir in [Path("/tmp/disabled_cache_orig"), Path("/tmp/disabled_cache_enh")]:
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
    
    # Summary
    print("\n" + "=" * 50)
    print("🏆 FINAL RESULTS")
    print("=" * 50)
    
    if results:
        total_improvement = sum(r["improvement"] for r in results) / len(results)
        
        print(f"{'Test':<20} {'Orig(s)':<10} {'Enh(s)':<10} {'Improve':<10}")
        print("-" * 50)
        
        for result in results:
            print(f"{result['name']:<20} {result['orig_time']:<10.3f} {result['enh_time']:<10.3f} {result['improvement']:<+10.1f}%")
        
        print(f"\n🎯 Average improvement: {total_improvement:+.1f}%")
        
        if total_improvement > 15:
            print("🏆 Significant performance improvement achieved!")
        elif total_improvement > 5:
            print("✅ Good performance improvement")
        elif total_improvement > -5:
            print("➡️ Performance is similar")
        else:
            print("⚠️ Performance degraded - needs optimization")
    
    print("\n🏁 Final test complete!")


if __name__ == "__main__":
    run_final_performance_test()
