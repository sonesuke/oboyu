"""MCP server integration for Oboyu.

This package provides MCP (Model Context Protocol) server functionality,
allowing AI assistants to interact with Oboyu's Japanese-enhanced semantic search.
"""

from oboyu.mcp.context import db_path_global, mcp
from oboyu.mcp.server import get_index_info, get_indexer, index_directory, search

__all__ = ["context", "server", "mcp", "db_path_global", "get_indexer", "search", "get_index_info", "index_directory"]
