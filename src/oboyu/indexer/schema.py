"""Legacy import compatibility for schema module."""

# Re-export from the new location
from oboyu.indexer.storage.schema import SCHEMA_MIGRATIONS, DatabaseSchema

__all__ = ["SCHEMA_MIGRATIONS", "DatabaseSchema"]
