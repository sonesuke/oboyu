"""MCP server implementation for Oboyu.

This module provides a FastMCP-based server that exposes Oboyu's search capabilities
through the Model Context Protocol (MCP), allowing AI assistants to interact with
the system's Japanese-enhanced semantic search.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from oboyu.common.paths import DEFAULT_DB_PATH
from oboyu.common.types import SearchFilters
from oboyu.indexer import Indexer
from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.config.model_config import ModelConfig
from oboyu.indexer.config.processing_config import ProcessingConfig
from oboyu.indexer.config.search_config import SearchConfig
from oboyu.mcp.context import db_path_global, mcp
from oboyu.retriever.retriever import Retriever
from oboyu.retriever.search.snippet_processor import SnippetProcessor
from oboyu.retriever.search.snippet_types import SnippetConfig

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

    # Create indexer configuration with minimal settings
    processing_config = ProcessingConfig(db_path=Path(actual_db_path))
    indexer_config = IndexerConfig(model=ModelConfig(), search=SearchConfig(), processing=processing_config)

    # Create and return indexer
    return Indexer(config=indexer_config)


def get_retriever(db_path: Optional[str] = None) -> Retriever:
    """Create and initialize a Retriever instance.

    Args:
        db_path: Optional path to the database file

    Returns:
        Initialized Retriever instance

    """
    # Use supplied path, global variable, or default (in that order)
    actual_db_path = db_path or db_path_global.value or str(DEFAULT_DB_PATH)

    # Create retriever configuration with minimal settings
    processing_config = ProcessingConfig(db_path=Path(actual_db_path))
    retriever_config = IndexerConfig(model=ModelConfig(), search=SearchConfig(), processing=processing_config)

    # Create and return retriever
    return Retriever(config=retriever_config)


@mcp.tool()
def search(
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    language: Optional[str] = None,
    db_path: Optional[str] = None,
    snippet_config: Optional[Dict[str, object]] = None,
    filters: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Execute high-precision semantic search with Japanese language optimization.
    
    🔍 Search Mode Optimization Guide:
    • vector: Conceptual search, semantic similarity, "explain about..." queries
    • bm25: Exact keyword matching, technical terms, API names, function names
    • hybrid: Balanced approach (recommended), optimal for general-purpose search
    
    📋 Parameters:
        query (str): Search query text (Japanese, English, or mixed supported)
            Examples: "machine learning algorithms", "機械学習アルゴリズム", "REST API design"
            
        mode (str): Search algorithm mode
            • "vector": Semantic similarity focus (best for conceptual queries)
            • "bm25": Keyword matching focus (best for technical searches)
            • "hybrid": Balanced combination (recommended for general use)
            
        top_k (int, 1-100): Number of results to return (recommended: 5-10)
            • 1-5: Highly curated results
            • 6-15: Broader candidate pool
            • 16-100: Comprehensive search coverage
            
        language (str, optional): Language filter for results
            • "ja": Japanese content only
            • "en": English content only
            • None: All languages (auto-detected)
            
        db_path (str, optional): Path to database file (uses default if not specified)
            
        snippet_config (dict, optional): Configuration for snippet generation
            • max_length: Maximum snippet length in characters
            • context_window: Characters around matching terms
            • highlight_terms: Whether to highlight matching terms
            
        filters (dict, optional): Filters for search results
            • date_range: Filter by document modification date
            • path_patterns: Include/exclude specific file paths
            • file_types: Filter by file extensions
    
    📤 Returns:
        Dictionary containing search results and statistics:
        • results: List of matching documents with title, content, uri, score, language, metadata
        • stats: Search statistics including count, query, and language filter
        
    💡 Usage Examples & Best Practices:
        # Japanese conceptual search (recommended: vector mode)
        search("機械学習の基本的な考え方", mode="vector")
        
        # Technical term search (recommended: bm25 mode)
        search("pandas DataFrame merge", mode="bm25")
        
        # General-purpose search (recommended: hybrid mode, default)
        search("project management best practices")
        
        # Language-specific search
        search("Python async programming", language="en", top_k=10)
        
        # Japanese-English mixed query
        search("Pythonでの非同期処理の実装", mode="hybrid")
        
        # Search with snippet customization
        search("API documentation", snippet_config={"max_length": 200, "highlight_terms": True})
        
        # Filtered search by file type and date
        search("configuration", filters={"file_types": [".py", ".json"], "path_patterns": ["src/**"]})
    
    🎯 Optimization Tips:
        • 2-5 words per query typically yield best results
        • Combine technical terms with general concepts
        • Japanese particles like "について" and "に関して" are well-supported
        • Try vector mode if too few results
        • Try bm25 mode if results lack precision
        • Mixed Japanese-English queries work naturally
        • For Japanese content, vector mode often improves conceptual understanding
        
    ❌ Troubleshooting:
        • No results → Use broader keywords / switch to vector mode
        • Low relevance → Use more specific terms / switch to bm25 mode
        • Slow performance → Reduce top_k / use language filter
        • Japanese text issues → Ensure proper encoding (auto-detected)
        
    🌏 Language Support:
        • Japanese: Full morphological analysis with MeCab
        • English: Advanced tokenization and stemming
        • Mixed queries: Automatic language detection per term
        • Encoding: Auto-detection for UTF-8, Shift-JIS, EUC-JP

    """
    try:
        # Initialize retriever
        retriever = get_retriever(db_path)
        
        # Parse filters if provided
        search_filters = None
        if filters:
            try:
                search_filters = SearchFilters.from_dict(filters)
            except Exception as e:
                logger.warning(f"Invalid filters: {e}, proceeding without filters")
                search_filters = None

        # Execute search with specified mode
        results = retriever.search(query, limit=top_k, mode=mode, language_filter=language, filters=search_filters)

        # Initialize snippet processor if config provided
        snippet_processor = None
        if snippet_config:
            try:
                # Convert dict to SnippetConfig with proper type handling
                config_dict: Dict[str, Any] = snippet_config
                config = SnippetConfig(**config_dict)
                snippet_processor = SnippetProcessor(config)
            except Exception as e:
                logger.warning(f"Invalid snippet_config: {e}, using default content")

        # Format results for MCP output
        formatted_results = []
        for result in results:
            # Process content with snippet processor if available
            content = result.content
            if snippet_processor:
                content = snippet_processor.generate_snippet(result.content, query, result.score)
            
            formatted_results.append(
                {
                    "title": result.title,
                    "content": content,
                    "uri": f"file://{result.path}",
                    "score": result.score,
                    "language": result.language,
                    "metadata": result.metadata,
                }
            )

        # Return results and statistics
        return {"results": formatted_results, "stats": {"count": len(formatted_results), "query": query, "language_filter": language or "none"}}
    except Exception as e:
        logger.error(f"Error executing search: {str(e)}")
        # Return structured error information
        return {
            "error": str(e),
            "error_type": "search_error",
            "results": [],
            "stats": {"count": 0, "query": query, "language_filter": language or "none"}
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
            return {"error": f"Directory does not exist or is not a directory: {directory_path}", "success": False, "documents_indexed": 0, "chunks_indexed": 0}

        # Initialize indexer
        indexer = get_indexer(db_path)

        # Perform indexing - NewIndexer doesn't have index_directory method
        # Use the crawler directly
        from oboyu.crawler.crawler import Crawler

        crawler = Crawler()
        crawler_results = crawler.crawl(directory)
        result = indexer.index_documents(crawler_results)
        chunks_indexed = result.get("indexed_chunks", 0)
        files_processed = result.get("total_documents", 0)

        # Return results
        return {
            "success": True,
            "directory": str(directory.absolute()),
            "documents_indexed": files_processed,
            "chunks_indexed": chunks_indexed,
            "db_path": indexer.config.db_path,
        }
    except Exception as e:
        logger.error(f"Error indexing directory: {str(e)}")
        return {"error": str(e), "success": False, "documents_indexed": 0, "chunks_indexed": 0}




@mcp.tool()
def get_index_info(db_path: Optional[str] = None) -> Dict[str, object]:
    """Return information about the index database.

    Args:
        db_path: Optional path to the database file

    Returns:
        Dictionary containing index statistics and metadata

    """
    try:
        # Create retriever to get stats (since stats include search-related info)
        retriever = get_retriever(db_path)
        
        # Query database for statistics
        db_stats = retriever.get_stats()

        # Return formatted statistics
        return {
            "document_count": db_stats.get("indexed_paths", 0),
            "chunk_count": db_stats.get("total_chunks", 0),
            "languages": ["unknown"],  # Not available in NewIndexer stats yet
            "embedding_model": db_stats.get("embedding_model", "unknown"),
            "db_path": str(retriever.config.processing.db_path) if retriever.config.processing else "unknown",
            "last_updated": "unknown",  # Not available in NewIndexer stats yet
        }
    except Exception as e:
        logger.error(f"Error retrieving index info: {str(e)}")
        # Return error information
        return {"error": str(e), "document_count": 0, "chunk_count": 0, "languages": [], "db_path": db_path or str(DEFAULT_DB_PATH)}
