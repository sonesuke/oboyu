"""Type-safe query builder for Oboyu indexer database operations.

This module provides backward compatibility by re-exporting all query functionality
from the consolidated queries module. The original QueryBuilder class is now implemented
by delegating to specialized query classes.

Key features:
- Type-safe query construction with parameter binding
- Support for all common database operations (CRUD)
- Specialized methods for vector and BM25 search
- Proper handling of complex data types (JSON, arrays, vectors)
- Built-in validation and error handling
"""

# Re-export everything from the consolidated queries for backward compatibility
from .consolidated_queries import *  # noqa: F403,F401
