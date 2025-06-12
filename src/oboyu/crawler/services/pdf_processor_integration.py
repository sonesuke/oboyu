"""Integration layer for optimized PDF processor with existing content extractor."""

import time
from pathlib import Path
from typing import Any, Dict, Tuple

from .optimized_pdf_processor import OptimizedPDFProcessor


class PDFProcessorIntegration:
    """Integration wrapper for the optimized PDF processor."""
    
    def __init__(self, max_file_size: int = 50 * 1024 * 1024) -> None:
        """Initialize integration wrapper.
        
        Args:
            max_file_size: Maximum file size in bytes

        """
        self.processor = OptimizedPDFProcessor(max_file_size=max_file_size)
        
    def extract_pdf_file(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract PDF with optimized processing strategies.
        
        This method can replace the existing _extract_pdf_file method
        in content_extractor.py for improved performance.
        """
        return self.processor.extract_pdf(file_path)


# Monkey patch integration for existing ContentExtractor
def integrate_optimized_processor() -> None:
    """Replace the existing PDF processor in ContentExtractor with the optimized version.
    
    Usage:
        from oboyu.crawler.services.pdf_processor_integration import integrate_optimized_processor
        integrate_optimized_processor()
    """
    try:
        from .content_extractor import ContentExtractor
        
        # Store original method
        ContentExtractor._original_extract_pdf_file = ContentExtractor._extract_pdf_file
        
        # Create optimized processor instance
        optimized_processor = OptimizedPDFProcessor()
        
        def optimized_extract_pdf_file(self: ContentExtractor, file_path: Path) -> Tuple[str, Dict[str, Any]]:
            """Optimized PDF extraction method."""
            return optimized_processor.extract_pdf(file_path)
        
        # Replace method
        ContentExtractor._extract_pdf_file = optimized_extract_pdf_file
        
        print("✅ Optimized PDF processor integrated successfully")
        
    except ImportError:
        print("❌ Could not integrate optimized processor - ContentExtractor not found")


# Performance comparison utility
class PDFPerformanceComparator:
    """Utility to compare performance between old and new processors."""
    
    def __init__(self) -> None:
        """Initialize performance comparator."""
        self.optimized = OptimizedPDFProcessor()
        
    def compare_processing(self, file_path: Path) -> Dict[str, Any]:
        """Compare performance between original and optimized processors."""
        try:
            from .content_extractor import ContentExtractor
            original_extractor = ContentExtractor()
        except ImportError:
            return {"error": "Could not import original ContentExtractor"}
        
        results = {}
        
        # Test optimized processor
        print(f"Testing optimized processor on {file_path.name}...")
        start_time = time.time()
        try:
            opt_content, opt_metadata = self.optimized.extract_pdf(file_path)
            opt_time = time.time() - start_time
            results["optimized"] = {
                "time": opt_time,
                "content_length": len(opt_content),
                "pages": opt_metadata.get("total_pages", 0),
                "strategy": opt_metadata.get("strategy_used", "unknown"),
                "success": True
            }
        except Exception as e:
            results["optimized"] = {
                "error": str(e),
                "success": False
            }
        
        # Test original processor
        print(f"Testing original processor on {file_path.name}...")
        start_time = time.time()
        try:
            orig_content, orig_metadata = original_extractor._extract_pdf_file(file_path)
            orig_time = time.time() - start_time
            results["original"] = {
                "time": orig_time,
                "content_length": len(orig_content),
                "pages": orig_metadata.get("total_pages", 0),
                "success": True
            }
        except Exception as e:
            results["original"] = {
                "error": str(e),
                "success": False
            }
        
        # Calculate improvement
        if results["optimized"]["success"] and results["original"]["success"]:
            speedup = results["original"]["time"] / results["optimized"]["time"]
            results["improvement"] = {
                "speedup": speedup,
                "time_saved": results["original"]["time"] - results["optimized"]["time"],
                "strategy_used": results["optimized"]["strategy"]
            }
            
            print("Performance comparison results:")
            print(f"  Original: {results['original']['time']:.2f}s")
            print(f"  Optimized: {results['optimized']['time']:.2f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Strategy: {results['optimized']['strategy']}")
        
        return results


# Example usage and testing
def test_optimized_processor() -> None:
    """Test function to demonstrate the optimized processor."""
    test_files = [
        "/Users/sonesuke/oboyu/test_225/nintendo_ar2024.pdf",        # Small/many pages
        "/Users/sonesuke/oboyu/test_225/daiichisankyo_ar2024_jp.pdf", # Medium
        "/Users/sonesuke/oboyu/test_225/fastretailing_ar2024_jp.pdf", # Large/slow
    ]
    
    processor = OptimizedPDFProcessor()
    
    for file_path_str in test_files:
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"Test file not found: {file_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing: {file_path.name}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            content, metadata = processor.extract_pdf(file_path)
            total_time = time.time() - start_time
            
            print("✅ Success!")
            print(f"   Content length: {len(content):,} characters")
            print(f"   Pages: {metadata.get('total_pages', 'unknown')}")
            print(f"   Strategy: {metadata.get('strategy_used', 'unknown')}")
            print(f"   Processing time: {metadata.get('processing_time', total_time):.2f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_optimized_processor()
