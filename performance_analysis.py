#!/usr/bin/env python3
"""Quick PDF performance analysis script."""

import cProfile
import io
import pstats
import tempfile
import time
from pathlib import Path

from oboyu.crawler.services.optimized_pdf_processor import OptimizedPDFProcessor
from tests.fixtures.pdf.create_test_pdfs import create_large_pdf


def analyze_pdf_performance():
    """Analyze PDF processing performance."""
    print("🔍 PDF Performance Analysis Starting...")
    
    processor = OptimizedPDFProcessor()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Test different PDF sizes
        test_cases = [
            ("small", 5, 100),      # 5 pages, 100 chars per page
            ("medium", 50, 500),    # 50 pages, 500 chars per page
            ("large", 150, 1000),   # 150 pages, 1000 chars per page
        ]
        
        results = []
        
        for name, pages, content_per_page in test_cases:
            pdf_path = tmp_path / f"{name}.pdf"
            
            print(f"\n📄 Creating {name} PDF ({pages} pages)...")
            create_large_pdf(pdf_path, pages, content_per_page)
            
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"File size: {file_size_mb:.2f} MB")
            
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
            
            # Get top time-consuming functions
            stats_buffer = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_buffer)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 functions
            
            profile_data = stats_buffer.getvalue()
            
            result = {
                'name': name,
                'file_size_mb': file_size_mb,
                'pages': pages,
                'processing_time': processing_time,
                'throughput_mb_per_sec': throughput,
                'content_length': len(content),
                'pages_processed': metadata.get('extracted_pages', 0),
                'strategy': metadata.get('strategy_used', 'unknown'),
                'success': success,
                'error': error,
                'profile_data': profile_data
            }
            
            results.append(result)
            
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            print(f"🚀 Throughput: {throughput:.2f} MB/s")
            print(f"📊 Strategy: {result['strategy']}")
            print(f"📝 Content: {len(content):,} characters")
            print(f"✅ Pages processed: {result['pages_processed']}/{pages}")
            
            if not success:
                print(f"❌ Error: {error}")
        
        # Analysis summary
        print("\n" + "="*60)
        print("📈 PERFORMANCE ANALYSIS SUMMARY")
        print("="*60)
        
        for result in results:
            print(f"\n{result['name'].upper()} PDF:")
            print(f"  Size: {result['file_size_mb']:.2f} MB ({result['pages']} pages)")
            print(f"  Time: {result['processing_time']:.2f}s")
            print(f"  Throughput: {result['throughput_mb_per_sec']:.2f} MB/s")
            print(f"  Strategy: {result['strategy']}")
            
            # Show top bottlenecks from profiling
            lines = result['profile_data'].split('\n')
            print("  Top functions by time:")
            for line in lines[5:15]:  # Skip header, show top 10
                if line.strip() and 'function calls' not in line:
                    print(f"    {line}")
        
        # Performance insights
        print("\n🎯 PERFORMANCE INSIGHTS:")
        
        # Check if throughput decreases with size
        throughputs = [r['throughput_mb_per_sec'] for r in results if r['success']]
        if len(throughputs) >= 2:
            if throughputs[-1] < throughputs[0]:
                print("⚠️  Throughput decreases with larger files - potential bottleneck")
            else:
                print("✅ Throughput scales well with file size")
        
        # Check processing times
        for result in results:
            if result['success']:
                time_per_page = result['processing_time'] / result['pages']
                if time_per_page > 0.1:  # More than 100ms per page
                    print(f"⚠️  {result['name']} PDF: {time_per_page:.3f}s per page (may be slow)")
                else:
                    print(f"✅ {result['name']} PDF: {time_per_page:.3f}s per page (good)")
        
        # Recommendations
        print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        # Analyze common bottlenecks from profiling data
        all_profile_data = '\n'.join(r['profile_data'] for r in results)
        
        if 'extract_text' in all_profile_data:
            print("🔧 Text extraction is a major bottleneck - consider caching or parallel processing")
        
        if 'PdfReader' in all_profile_data:
            print("🔧 PDF parsing overhead detected - consider file streaming or chunked processing")
        
        if 'ThreadPoolExecutor' in all_profile_data:
            print("✅ Parallel processing is being used effectively")
        
        # Memory recommendations
        largest_file = max(results, key=lambda x: x['file_size_mb'])
        if largest_file['file_size_mb'] > 20:
            print("🔧 Large files detected - ensure streaming strategy is working")
        
        print("\n🏁 Analysis complete!")
        return results


if __name__ == "__main__":
    analyze_pdf_performance()
