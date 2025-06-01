"""Core indexing operations for the CLI."""

import time
from pathlib import Path
from typing import List, Optional

from oboyu.cli.base import BaseCommand
from oboyu.cli.index_config import build_indexer_config, create_indexer_config
from oboyu.cli.progress import create_indexer_progress_callback
from oboyu.indexer import Indexer


def execute_indexing_operation(
    base_command: BaseCommand,
    directories: List[Path],
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    embedding_model: Optional[str],
    db_path: Optional[Path],
    change_detection: Optional[str],
    cleanup_deleted: Optional[bool],
    verify_integrity: bool,
    quiet_progress: bool,
    force: bool,
    max_depth: Optional[int],
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]],
) -> None:
    """Execute the main indexing operation."""
    config_manager = base_command.get_config_manager()
    
    indexer_config_dict = create_indexer_config(
        config_manager,
        chunk_size,
        chunk_overlap,
        embedding_model,
        db_path,
    )

    base_command.print_database_path(indexer_config_dict["db_path"])
    indexer_config = build_indexer_config(indexer_config_dict)

    with base_command.logger.live_display():
        init_op = base_command.logger.start_operation("Initializing Oboyu indexer...")
        model_name = indexer_config_dict.get("embedding_model", "cl-nagoya/ruri-v3-30m")
        load_op = base_command.logger.start_operation(f"Loading embedding model ({model_name})...")

        indexer = Indexer(config=indexer_config)

        base_command.logger.complete_operation(load_op)
        base_command.logger.complete_operation(init_op)

        total_chunks = 0
        total_files = 0
        start_time = time.time()

        for directory in directories:
            scan_op_id = base_command.logger.start_operation(f"Scanning directory {directory}...", expandable=False)

            if quiet_progress:
                indexer_progress_callback = None
            else:
                indexer_progress_callback = create_indexer_progress_callback(base_command.logger, scan_op_id)

            _detection_strategy = change_detection or "smart"
            if verify_integrity:
                _detection_strategy = "hash"

            _should_cleanup = cleanup_deleted if cleanup_deleted is not None else True

            from oboyu.crawler.crawler import Crawler

            crawler = Crawler(
                depth=max_depth or 10,
                include_patterns=include_patterns or ["*.txt", "*.md", "*.html", "*.py", "*.java"],
                exclude_patterns=exclude_patterns or ["*/node_modules/*", "*/venv/*"],
                max_workers=4,
                respect_gitignore=True,
            )

            crawler_results = crawler.crawl(directory, progress_callback=indexer_progress_callback)
            result = indexer.index_documents(crawler_results, progress_callback=indexer_progress_callback)
            chunks_indexed, files_processed = result.get("indexed_chunks", 0), result.get("total_documents", 0)

            # This section needs proper diff_stats implementation
            # For now, using simple summary
            summary_op = base_command.logger.start_operation(f"Indexed {chunks_indexed} chunks from {files_processed} documents")
            base_command.logger.complete_operation(summary_op)

            total_chunks += chunks_indexed
            total_files += files_processed

        indexer.close()

    elapsed_time = time.time() - start_time
    base_command.console.print(f"\nIndexed {total_files} files ({total_chunks} chunks) in {elapsed_time:.1f}s")
