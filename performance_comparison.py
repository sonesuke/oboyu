#!/usr/bin/env python3
"""Compare original vs enhanced PDF processor performance."""

import time
from pathlib import Path

from oboyu.crawler.services.enhanced_pdf_processor import EnhancedPDFProcessor
from oboyu.crawler.services.optimized_pdf_processor import OptimizedPDFProcessor


def compare_processors():
    """Compare original vs enhanced PDF processor performance."""
    print("🔬 PDF Processor Performance Comparison")
    print("=" * 60)
    
    # Initialize processors
    original_processor = OptimizedPDFProcessor()
    enhanced_processor = EnhancedPDFProcessor()
    
    # Test files
    test_dir = Path("tests/fixtures/pdf/performance_tests")
    test_files = [
        ("small_reliable.pdf", "Small (5 pages)"),
        ("medium_reliable.pdf", "Medium (25 pages)"),
        ("large_reliable.pdf", "Large (100 pages)"),
        ("xl_reliable.pdf", "XL (200 pages)"),
    ]
    
    results = {"original": [], "enhanced": []}
    
    for filename, description in test_files:
        pdf_path = test_dir / filename
        
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            continue
        
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"\n📄 Testing {description} ({file_size_mb:.3f} MB)")
        print("-" * 50)
        
        # Test original processor
        print("🔧 Original Processor:")
        try:
            start_time = time.time()
            content_orig, metadata_orig = original_processor.extract_pdf(pdf_path)
            time_orig = time.time() - start_time
            
            throughput_orig = file_size_mb / time_orig if time_orig > 0 else 0
            pages_orig = metadata_orig.get('extracted_pages', 0)
            total_pages_orig = metadata_orig.get('total_pages', 0)
            
            print(f"   ⏱️  Time: {time_orig:.3f}s")
            print(f"   🚀 Throughput: {throughput_orig:.2f} MB/s")
            print(f"   📝 Content: {len(content_orig):,} chars")
            print(f"   ✅ Pages: {pages_orig}/{total_pages_orig}")
            
            results["original"].append({
                "name": description,
                "time": time_orig,
                "throughput": throughput_orig,
                "content_length": len(content_orig),
                "pages_extracted": pages_orig,
                "total_pages": total_pages_orig,
                "success": pages_orig == total_pages_orig
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results["original"].append({
                "name": description,
                "time": float('inf'),
                "throughput": 0,
                "content_length": 0,
                "pages_extracted": 0,
                "total_pages": 0,
                "success": False,
                "error": str(e)
            })
        
        # Test enhanced processor
        print("\n⚡ Enhanced Processor:")
        try:
            start_time = time.time()
            content_enh, metadata_enh = enhanced_processor.extract_pdf(pdf_path)
            time_enh = time.time() - start_time
            
            throughput_enh = file_size_mb / time_enh if time_enh > 0 else 0
            pages_enh = metadata_enh.get('extracted_pages', 0)
            total_pages_enh = metadata_enh.get('total_pages', 0)
            
            print(f"   ⏱️  Time: {time_enh:.3f}s")
            print(f"   🚀 Throughput: {throughput_enh:.2f} MB/s")
            print(f"   📝 Content: {len(content_enh):,} chars")
            print(f"   ✅ Pages: {pages_enh}/{total_pages_enh}")
            
            results["enhanced"].append({
                "name": description,
                "time": time_enh,
                "throughput": throughput_enh,
                "content_length": len(content_enh),
                "pages_extracted": pages_enh,
                "total_pages": total_pages_enh,
                "success": pages_enh == total_pages_enh
            })
            
            # Calculate improvement
            if results["original"][-1]["time"] != float('inf'):
                time_improvement = (time_orig - time_enh) / time_orig * 100
                throughput_improvement = (throughput_enh - throughput_orig) / throughput_orig * 100 if throughput_orig > 0 else 0
                
                print("\n📈 Improvements:")
                print(f"   ⚡ Time: {time_improvement:+.1f}% ({'faster' if time_improvement > 0 else 'slower'})")
                print(f"   🚀 Throughput: {throughput_improvement:+.1f}%")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results["enhanced"].append({
                "name": description,
                "time": float('inf'),
                "throughput": 0,
                "content_length": 0,
                "pages_extracted": 0,
                "total_pages": 0,
                "success": False,
                "error": str(e)
            })
    
    # Overall comparison
    print("\n" + "=" * 60)
    print("📊 OVERALL COMPARISON")
    print("=" * 60)
    
    print(f"{'Test':<20} {'Original(s)':<12} {'Enhanced(s)':<12} {'Improvement':<12} {'Status':<10}")
    print("-" * 78)
    
    total_time_orig = 0
    total_time_enh = 0
    improvements = []
    
    for i, (orig, enh) in enumerate(zip(results["original"], results["enhanced"])):
        if orig["time"] != float('inf') and enh["time"] != float('inf'):
            improvement = (orig["time"] - enh["time"]) / orig["time"] * 100
            improvements.append(improvement)
            total_time_orig += orig["time"]
            total_time_enh += enh["time"]
            
            status = "✅ Better" if improvement > 5 else "➡️ Similar" if improvement > -5 else "⚠️ Slower"
            
            print(f"{orig['name']:<20} {orig['time']:<12.3f} {enh['time']:<12.3f} {improvement:<+12.1f}% {status:<10}")
        else:
            print(f"{orig['name']:<20} {'ERROR':<12} {'ERROR':<12} {'N/A':<12} {'❌ Error':<10}")
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        total_improvement = (total_time_orig - total_time_enh) / total_time_orig * 100
        
        print("\n🎯 SUMMARY:")
        print(f"   Average improvement per test: {avg_improvement:+.1f}%")
        print(f"   Total time improvement: {total_improvement:+.1f}%")
        print(f"   Original total time: {total_time_orig:.3f}s")
        print(f"   Enhanced total time: {total_time_enh:.3f}s")
        
        if avg_improvement > 10:
            print("   🏆 Significant performance improvement achieved!")
        elif avg_improvement > 0:
            print("   ✅ Performance improvement achieved")
        else:
            print("   ⚠️  No significant improvement (needs further optimization)")
    
    print("\n🏁 Comparison complete!")
    return results


if __name__ == "__main__":
    compare_processors()
