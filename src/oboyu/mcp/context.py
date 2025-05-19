"""Global context for MCP server.

This module contains shared global variables and configuration
that can be accessed across the MCP server implementation.
"""

from dataclasses import dataclass
from typing import Optional

from mcp.server.fastmcp import FastMCP


@dataclass
class DBPathGlobal:
    """Global storage for database path."""

    value: Optional[str] = None


# Global variable to store the current database path
db_path_global = DBPathGlobal()

# Create MCP server instance with descriptive name
mcp = FastMCP(name="Oboyu - Japanese-Enhanced Semantic Search")
