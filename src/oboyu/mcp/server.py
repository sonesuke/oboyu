"""MCP server implementation for Oboyu.

This module provides a FastMCP-based server that exposes Oboyu's search capabilities
through the Model Context Protocol (MCP), allowing AI assistants to interact with
the system's Japanese-enhanced semantic search.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from oboyu.common.paths import DEFAULT_DB_PATH
from oboyu.indexer.config import IndexerConfig
from oboyu.indexer.indexer import Indexer
from oboyu.mcp.context import db_path_global, mcp

# Configure logging
logger = logging.getLogger(__name__)


def get_indexer(db_path: Optional[str] = None) -> Indexer:
    """Create and initialize an Indexer instance.

    Args:
        db_path: Optional path to the database file

    Returns:
        Initialized Indexer instance

    """
    # Use supplied path, global variable, or default (in that order)
    actual_db_path = db_path or db_path_global.value or str(DEFAULT_DB_PATH)

    # Create indexer configuration
    config_dict = {"indexer": {"db_path": actual_db_path}}
    indexer_config = IndexerConfig(config_dict=config_dict)

    # Create and return indexer
    return Indexer(config=indexer_config)


@mcp.tool()
def search(
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    language: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, object]:
    """Execute a semantic search query and return relevant documents.

    Args:
        query: Search query
        mode: Search mode (vector, bm25, hybrid)
        top_k: Maximum number of results to return
        language: Optional language filter (e.g., 'ja', 'en')
        db_path: Optional path to the database file

    Returns:
        Dictionary containing search results and statistics

    """
    try:
        # Initialize indexer
        indexer = get_indexer(db_path)

        # Execute search with specified mode
        results = indexer.search(query, limit=top_k, mode=mode, language=language)

        # Format results for MCP output
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.title,
                "content": result.content,
                "uri": f"file://{result.path}",
                "score": result.score,
                "language": result.language,
                "metadata": result.metadata
            })

        # Return results and statistics
        return {
            "results": formatted_results,
            "stats": {
                "count": len(formatted_results),
                "query": query,
                "language_filter": language or "none"
            }
        }
    except Exception as e:
        logger.error(f"Error executing search: {str(e)}")
        # Return error information
        return {
            "error": str(e),
            "results": [],
            "stats": {
                "count": 0,
                "query": query
            }
        }


@mcp.tool()
def index_directory(
    directory_path: str,
    incremental: bool = True,
    db_path: Optional[str] = None,
) -> Dict[str, object]:
    """Index documents in a directory.

    Args:
        directory_path: Path to the directory to index
        incremental: Only index new or changed files
        db_path: Optional path to the database file

    Returns:
        Dictionary containing indexing results

    """
    try:
        # Validate directory exists
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            return {
                "error": f"Directory does not exist or is not a directory: {directory_path}",
                "success": False,
                "documents_indexed": 0,
                "chunks_indexed": 0
            }

        # Initialize indexer
        indexer = get_indexer(db_path)

        # Perform indexing
        chunks_indexed, files_processed = indexer.index_directory(
            directory=directory,
            incremental=incremental
        )

        # Return results
        return {
            "success": True,
            "directory": str(directory.absolute()),
            "documents_indexed": files_processed,
            "chunks_indexed": chunks_indexed,
            "db_path": indexer.config.db_path
        }
    except Exception as e:
        logger.error(f"Error indexing directory: {str(e)}")
        return {
            "error": str(e),
            "success": False,
            "documents_indexed": 0,
            "chunks_indexed": 0
        }


@mcp.tool()
def clear_index(
    db_path: Optional[str] = None,
) -> Dict[str, object]:
    """Clear all data from the index database.

    Args:
        db_path: Optional path to the database file

    Returns:
        Dictionary containing operation results

    """
    try:
        # Initialize indexer
        indexer = get_indexer(db_path)

        # Clear the index
        indexer.clear_index()

        # Return success
        return {
            "success": True,
            "message": "Index database cleared successfully",
            "db_path": indexer.config.db_path
        }
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }


@mcp.tool()
def get_index_info(db_path: Optional[str] = None) -> Dict[str, object]:
    """Return information about the index database.

    Args:
        db_path: Optional path to the database file

    Returns:
        Dictionary containing index statistics and metadata

    """
    try:
        # Initialize indexer
        indexer = get_indexer(db_path)

        # Query database for statistics
        db_stats = indexer.database.get_statistics()

        # Return formatted statistics
        return {
            "document_count": db_stats.get("document_count", 0),
            "chunk_count": db_stats.get("chunk_count", 0),
            "languages": db_stats.get("languages", ["unknown"]),
            "embedding_model": db_stats.get("embedding_model", "unknown"),
            "db_path": db_stats.get("db_path", str(DEFAULT_DB_PATH)),
            "last_updated": db_stats.get("last_updated", "unknown")
        }
    except Exception as e:
        logger.error(f"Error retrieving index info: {str(e)}")
        # Return error information
        return {
            "error": str(e),
            "document_count": 0,
            "chunk_count": 0,
            "languages": [],
            "db_path": db_path or str(DEFAULT_DB_PATH)
        }
