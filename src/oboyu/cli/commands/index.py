"""Consolidated indexing command functionality."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from oboyu.cli.index_config import build_indexer_config, create_indexer_config
from oboyu.cli.progress import create_indexer_progress_callback
from oboyu.cli.services.console_manager import ConsoleManager
from oboyu.cli.services.indexer_factory import IndexerFactory
from oboyu.common.config import ConfigManager
from oboyu.crawler.config import load_default_config
from oboyu.crawler.crawler import Crawler
from oboyu.crawler.discovery import discover_documents
from oboyu.indexer import Indexer
from oboyu.indexer.config.indexer_config import IndexerConfig

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result of an indexing operation."""
    
    total_files: int
    total_chunks: int
    elapsed_time: float
    directories_processed: List[Path]


@dataclass
class StatusResult:
    """Result of a status check operation."""
    
    new_files: int
    modified_files: int
    deleted_files: int
    total_indexed: int
    directory: Path


@dataclass
class DiffResult:
    """Result of a diff operation."""
    
    new_files: List[str]
    modified_files: List[str]
    deleted_files: List[str]
    directory: Path


class IndexCommand:
    """Consolidated indexing command service combining indexing functionality."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        indexer_factory: Optional[IndexerFactory] = None,
        console_manager: Optional[ConsoleManager] = None,
    ) -> None:
        """Initialize the indexing command service.
        
        Args:
            config_manager: Configuration manager instance
            indexer_factory: Factory for creating indexer instances with progress
            console_manager: Console manager for progress display

        """
        self.config_manager = config_manager
        self.indexer_factory = indexer_factory
        self.console_manager = console_manager
    
    def create_indexer_config(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        db_path: Optional[Path] = None,
    ) -> IndexerConfig:
        """Create indexer configuration."""
        indexer_config_dict = create_indexer_config(
            self.config_manager,
            chunk_size,
            chunk_overlap,
            embedding_model,
            db_path,
        )
        return build_indexer_config(indexer_config_dict)
    
    def execute_indexing(
        self,
        directories: List[Path],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        db_path: Optional[Path] = None,
        change_detection: Optional[str] = None,
        cleanup_deleted: Optional[bool] = None,
        verify_integrity: bool = False,
        force: bool = False,
        max_depth: Optional[int] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> IndexingResult:
        """Execute indexing operation."""
        indexer_config = self.create_indexer_config(
            chunk_size, chunk_overlap, embedding_model, db_path
        )
        
        start_time = time.time()
        total_chunks = 0
        total_files = 0
        
        # Create indexer with progress display if factory is available
        if self.indexer_factory and self.console_manager:
            indexer = self.indexer_factory.create_indexer(indexer_config, self.console_manager)
        else:
            indexer = Indexer(config=indexer_config)
        
        try:
            for directory in directories:
                # Start scan operation for progress tracking
                scan_op_id = ""
                if self.console_manager:
                    scan_op_id = self.console_manager.logger.start_operation(f"Scanning directory {directory}...")
                elif progress_callback:
                    progress_callback(f"Scanning directory {directory}...")
                
                crawler = Crawler(
                    depth=max_depth or 10,
                    include_patterns=include_patterns or ["*.txt", "*.md", "*.html", "*.py", "*.java"],
                    exclude_patterns=exclude_patterns or ["*/node_modules/*", "*/venv/*"],
                    max_workers=4,
                    respect_gitignore=True,
                )
                
                crawler_results = crawler.crawl(directory)
                
                # Create indexer progress callback if console manager is available
                indexer_progress_callback = None
                if self.console_manager and scan_op_id:
                    indexer_progress_callback = create_indexer_progress_callback(
                        self.console_manager.logger,
                        scan_op_id
                    )
                
                try:
                    result = indexer.index_documents(crawler_results, indexer_progress_callback)
                    
                    chunks_indexed = result.get("indexed_chunks", 0)
                    files_processed = result.get("total_documents", 0)
                    
                    total_chunks += chunks_indexed
                    total_files += files_processed
                    
                    if progress_callback:
                        progress_callback(f"Indexed {chunks_indexed} chunks from {files_processed} documents")
                
                except RuntimeError as e:
                    # Check if this is a model loading error from our services
                    if "Failed to load" in str(e) and "model" in str(e):
                        error_msg = f"❌ Indexing failed due to model loading error:\n{str(e)}"
                        if progress_callback:
                            progress_callback(error_msg)
                        raise RuntimeError(error_msg) from e
                    else:
                        raise
        except Exception:
            # Ensure clean shutdown
            try:
                indexer.close()
            except Exception as cleanup_error:
                # Ignore errors during cleanup but log for debugging
                logger.debug(f"Error during indexer cleanup: {cleanup_error}")
            raise
        finally:
            indexer.close()
        
        elapsed_time = time.time() - start_time
        
        return IndexingResult(
            total_files=total_files,
            total_chunks=total_chunks,
            elapsed_time=elapsed_time,
            directories_processed=directories,
        )
    
    def get_status(
        self,
        directories: List[Path],
        db_path: Optional[Path] = None,
    ) -> List[StatusResult]:
        """Get indexing status for directories."""
        indexer_config_dict = create_indexer_config(
            self.config_manager,
            None,  # chunk_size
            None,  # chunk_overlap
            None,  # embedding_model
            db_path,
        )
        
        from oboyu.cli.index_config import build_status_indexer_config
        indexer_config = build_status_indexer_config(indexer_config_dict)
        indexer = Indexer(config=indexer_config)
        
        results = []
        
        try:
            crawler_config = load_default_config()
            
            for directory in directories:
                discovered_paths = discover_documents(
                    Path(directory),
                    patterns=crawler_config.include_patterns,
                    exclude_patterns=crawler_config.exclude_patterns,
                    max_depth=crawler_config.depth,
                )
                
                paths_only = [path for path, metadata in discovered_paths]
                changes = indexer.change_detector.detect_changes(paths_only, strategy="smart")
                stats = indexer.change_detector.get_processing_stats()
                
                results.append(StatusResult(
                    new_files=len(changes.new_files),
                    modified_files=len(changes.modified_files),
                    deleted_files=len(changes.deleted_files),
                    total_indexed=stats.get('completed', 0),
                    directory=directory,
                ))
        finally:
            indexer.close()
        
        return results
    
    def get_diff(
        self,
        directories: List[Path],
        db_path: Optional[Path] = None,
        change_detection: Optional[str] = None,
    ) -> List[DiffResult]:
        """Get diff of what would be updated."""
        indexer_config_dict = create_indexer_config(
            self.config_manager,
            None,  # chunk_size
            None,  # chunk_overlap
            None,  # embedding_model
            db_path,
        )
        
        from oboyu.cli.index_config import build_status_indexer_config
        indexer_config = build_status_indexer_config(indexer_config_dict)
        indexer = Indexer(config=indexer_config)
        
        results = []
        
        try:
            crawler_config = load_default_config()
            detection_strategy = change_detection or "smart"
            
            for directory in directories:
                discovered_paths = discover_documents(
                    Path(directory),
                    patterns=crawler_config.include_patterns,
                    exclude_patterns=crawler_config.exclude_patterns,
                    max_depth=crawler_config.depth,
                )
                
                paths_only = [path for path, metadata in discovered_paths]
                changes = indexer.change_detector.detect_changes(paths_only, strategy=detection_strategy)
                
                results.append(DiffResult(
                    new_files=[str(f) for f in sorted(changes.new_files)],
                    modified_files=[str(f) for f in sorted(changes.modified_files)],
                    deleted_files=[str(f) for f in sorted(changes.deleted_files)],
                    directory=directory,
                ))
        finally:
            indexer.close()
        
        return results
    
    def clear_index(self, db_path: Optional[Path] = None) -> None:
        """Clear the index database."""
        indexer_config = self.create_indexer_config(db_path=db_path)
        indexer = Indexer(config=indexer_config)
        
        try:
            indexer.clear_index()
        finally:
            indexer.close()
    
    def get_database_path(self, db_path: Optional[Path] = None) -> str:
        """Get the resolved database path."""
        indexer_config_dict = create_indexer_config(
            self.config_manager,
            None,  # chunk_size
            None,  # chunk_overlap
            None,  # embedding_model
            db_path,
        )
        return str(indexer_config_dict["db_path"])


# Legacy alias for backward compatibility
IndexingService = IndexCommand
