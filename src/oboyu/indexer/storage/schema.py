"""Database schema management for Oboyu indexer.

This module provides centralized database schema definitions, migrations,
and version management for the Oboyu indexer database.

The schema is organized into logical groups:
- Core tables: chunks, embeddings
- BM25 tables: vocabulary, inverted_index, document_stats, collection_stats
- Meta tables: schema_version (for migration tracking)

Key features:
- Centralized SQL definitions for maintainability
- Schema versioning for safe migrations
- Type-safe table and column definitions
- Support for both initial creation and migrations
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class TableDefinition:
    """Definition of a database table."""

    name: str
    sql: str
    indexes: List[str]
    dependencies: List[str]  # Tables this table depends on (for foreign keys)


@dataclass
class SchemaVersion:
    """Schema version information."""

    version: str
    description: str
    migration_sql: List[str]
    rollback_sql: List[str]


class DatabaseSchema:
    """Centralized database schema management.

    This class provides all table definitions, indexes, and migration logic
    for the Oboyu indexer database. All SQL statements are centralized here
    to improve maintainability and enable safe schema evolution.
    """

    # Current schema version
    CURRENT_VERSION = "1.0.0"

    def __init__(self, embedding_dimensions: int = 256) -> None:
        """Initialize schema with embedding dimensions.

        Args:
            embedding_dimensions: Vector dimensions for embeddings table

        """
        self.embedding_dimensions = embedding_dimensions

    def get_chunks_table(self) -> TableDefinition:
        """Get chunks table definition."""
        return TableDefinition(
            name="chunks",
            sql="""
                CREATE TABLE IF NOT EXISTS chunks (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,             -- Path to source file
                    title VARCHAR,            -- Document or chunk title
                    content TEXT,             -- Chunk text content
                    chunk_index INTEGER,      -- Position in original document
                    language VARCHAR,         -- Language code (e.g., 'ja', 'en')
                    created_at TIMESTAMP,     -- Creation timestamp
                    modified_at TIMESTAMP,    -- Last modification timestamp
                    metadata JSON             -- Additional metadata as JSON
                )
            """.strip(),
            indexes=[
                "CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks(created_at)",
            ],
            dependencies=[],
        )

    def get_embeddings_table(self) -> TableDefinition:
        """Get embeddings table definition."""
        return TableDefinition(
            name="embeddings",
            sql=f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id VARCHAR PRIMARY KEY,
                    chunk_id VARCHAR,         -- Related chunk ID
                    model VARCHAR,            -- Embedding model used
                    vector FLOAT[{self.embedding_dimensions}],  -- Vector with specific dimensions
                    created_at TIMESTAMP,     -- Embedding generation timestamp
                    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
                )
            """.strip(),
            indexes=[
                "CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id)",
                "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model)",
            ],
            dependencies=["chunks"],
        )

    def get_vocabulary_table(self) -> TableDefinition:
        """Get BM25 vocabulary table definition."""
        return TableDefinition(
            name="vocabulary",
            sql="""
                CREATE TABLE IF NOT EXISTS vocabulary (
                    term VARCHAR PRIMARY KEY,
                    document_frequency INTEGER NOT NULL,  -- Number of documents containing term
                    collection_frequency INTEGER NOT NULL -- Total occurrences across collection
                )
            """.strip(),
            indexes=[
                "CREATE INDEX IF NOT EXISTS idx_vocabulary_doc_freq ON vocabulary(document_frequency)",
            ],
            dependencies=[],
        )

    def get_inverted_index_table(self) -> TableDefinition:
        """Get BM25 inverted index table definition."""
        return TableDefinition(
            name="inverted_index",
            sql="""
                CREATE TABLE IF NOT EXISTS inverted_index (
                    term VARCHAR,
                    chunk_id VARCHAR,
                    term_frequency INTEGER NOT NULL,  -- Frequency of term in chunk
                    positions INTEGER[],              -- Array of token positions for phrase search
                    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
                )
            """.strip(),
            indexes=[
                "CREATE INDEX IF NOT EXISTS idx_inverted_index_term ON inverted_index(term)",
                "CREATE INDEX IF NOT EXISTS idx_inverted_index_chunk ON inverted_index(chunk_id)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_inverted_index_term_chunk ON inverted_index(term, chunk_id)",
            ],
            dependencies=["chunks"],
        )

    def get_document_stats_table(self) -> TableDefinition:
        """Get BM25 document statistics table definition."""
        return TableDefinition(
            name="document_stats",
            sql="""
                CREATE TABLE IF NOT EXISTS document_stats (
                    chunk_id VARCHAR PRIMARY KEY,
                    total_terms INTEGER NOT NULL,        -- Total number of terms in document
                    unique_terms INTEGER NOT NULL,       -- Number of unique terms
                    avg_term_frequency REAL NOT NULL,    -- Average term frequency
                    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
                )
            """.strip(),
            indexes=[],
            dependencies=["chunks"],
        )

    def get_collection_stats_table(self) -> TableDefinition:
        """Get BM25 collection statistics table definition."""
        return TableDefinition(
            name="collection_stats",
            sql="""
                CREATE TABLE IF NOT EXISTS collection_stats (
                    id INTEGER PRIMARY KEY DEFAULT 1,  -- Single row table
                    total_documents INTEGER NOT NULL,   -- Total number of documents
                    total_terms INTEGER NOT NULL,       -- Total terms across collection
                    avg_document_length REAL NOT NULL,  -- Average document length
                    last_updated TIMESTAMP NOT NULL     -- Last update timestamp
                )
            """.strip(),
            indexes=[],
            dependencies=[],
        )

    def get_file_metadata_table(self) -> TableDefinition:
        """Get file metadata table for tracking indexed files."""
        return TableDefinition(
            name="file_metadata",
            sql="""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    path VARCHAR PRIMARY KEY,
                    last_processed_at TIMESTAMP NOT NULL,
                    file_modified_at TIMESTAMP NOT NULL,
                    file_size BIGINT NOT NULL,
                    content_hash VARCHAR,                    -- SHA-256 hash of file content
                    chunk_count INTEGER NOT NULL DEFAULT 0,  -- Number of chunks created from file
                    processing_status VARCHAR DEFAULT 'completed',  -- 'completed', 'error', 'in_progress'
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """.strip(),
            indexes=[
                "CREATE INDEX IF NOT EXISTS idx_file_metadata_status ON file_metadata(processing_status)",
                "CREATE INDEX IF NOT EXISTS idx_file_metadata_modified ON file_metadata(file_modified_at)",
                "CREATE INDEX IF NOT EXISTS idx_file_metadata_processed ON file_metadata(last_processed_at)",
            ],
            dependencies=[],
        )

    def get_schema_version_table(self) -> TableDefinition:
        """Get schema version table for migration tracking."""
        return TableDefinition(
            name="schema_version",
            sql="""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version VARCHAR PRIMARY KEY,
                    description VARCHAR NOT NULL,
                    applied_at TIMESTAMP NOT NULL,
                    migration_checksum VARCHAR     -- Checksum of migration SQL for validation
                )
            """.strip(),
            indexes=[],
            dependencies=[],
        )

    def get_all_tables(self) -> List[TableDefinition]:
        """Get all table definitions in dependency order.

        Returns:
            List of table definitions ordered by dependencies

        """
        tables = [
            self.get_schema_version_table(),
            self.get_chunks_table(),
            self.get_embeddings_table(),
            self.get_vocabulary_table(),
            self.get_inverted_index_table(),
            self.get_document_stats_table(),
            self.get_collection_stats_table(),
            self.get_file_metadata_table(),
        ]

        # Sort by dependencies to ensure proper creation order
        return self._sort_by_dependencies(tables)

    def _sort_by_dependencies(self, tables: List[TableDefinition]) -> List[TableDefinition]:
        """Sort tables by their dependencies.

        Args:
            tables: List of table definitions

        Returns:
            Sorted list ensuring dependencies come first

        """
        # Create a mapping for quick lookup
        table_map = {t.name: t for t in tables}
        sorted_tables = []
        processed = set()

        def add_table(table: TableDefinition) -> None:
            if table.name in processed:
                return

            # First add all dependencies
            for dep_name in table.dependencies:
                if dep_name in table_map:
                    add_table(table_map[dep_name])

            # Then add this table
            sorted_tables.append(table)
            processed.add(table.name)

        # Process all tables
        for table in tables:
            add_table(table)

        return sorted_tables

    def get_hnsw_index_sql(self, ef_construction: int = 128, ef_search: int = 64, m: int = 16, m0: Optional[int] = None) -> str:
        """Get HNSW index creation SQL.

        Args:
            ef_construction: Index construction parameter
            ef_search: Search time parameter
            m: Number of bidirectional links
            m0: Level-0 connections (defaults to 2*m)

        Returns:
            SQL for creating HNSW index

        """
        m0_val = m0 if m0 is not None else 2 * m

        return f"""
            CREATE INDEX IF NOT EXISTS vector_idx ON embeddings
            USING HNSW (vector)
            WITH (
                metric = 'cosine',
                ef_construction = {ef_construction},
                ef_search = {ef_search},
                m = {m},
                m0 = {m0_val}
            )
        """.strip()

    def get_initial_schema_version_data(self) -> Tuple[str, str, str, Optional[str]]:
        """Get initial schema version record data.

        Returns:
            Tuple of (version, description, applied_at, checksum)

        """
        return (
            self.CURRENT_VERSION,
            "Initial schema creation",
            datetime.now().isoformat(),
            None,  # No checksum for initial version
        )

    def get_drop_all_tables_sql(self) -> List[str]:
        """Get SQL to drop all tables in correct order.

        Returns:
            List of DROP TABLE statements in dependency-safe order

        """
        # Reverse dependency order for safe dropping
        tables = self.get_all_tables()
        return [f"DROP TABLE IF EXISTS {table.name}" for table in reversed(tables)]

    def validate_schema_consistency(self) -> List[str]:
        """Validate schema consistency and return any issues.

        Returns:
            List of validation error messages (empty if valid)

        """
        issues = []
        tables = self.get_all_tables()
        table_names = {t.name for t in tables}

        # Check dependency references
        for table in tables:
            for dep in table.dependencies:
                if dep not in table_names:
                    issues.append(f"Table '{table.name}' depends on unknown table '{dep}'")

        # Check for circular dependencies
        def has_circular_deps(table_name: str, visited: set[str], stack: set[str]) -> bool:
            if table_name in stack:
                return True
            if table_name in visited:
                return False

            visited.add(table_name)
            stack.add(table_name)

            table = next((t for t in tables if t.name == table_name), None)
            if table:
                for dep in table.dependencies:
                    if has_circular_deps(dep, visited, stack):
                        return True

            stack.remove(table_name)
            return False

        visited_global: set[str] = set()
        for table in tables:
            if table.name not in visited_global:
                if has_circular_deps(table.name, visited_global, set()):
                    issues.append(f"Circular dependency detected involving table '{table.name}'")

        return issues


# Schema migration definitions for future versions
SCHEMA_MIGRATIONS: Dict[str, SchemaVersion] = {
    "1.1.0": SchemaVersion(
        version="1.1.0",
        description="Add file metadata tracking for differential updates",
        migration_sql=[
            """
            CREATE TABLE IF NOT EXISTS file_metadata (
                path VARCHAR PRIMARY KEY,
                last_processed_at TIMESTAMP NOT NULL,
                file_modified_at TIMESTAMP NOT NULL,
                file_size BIGINT NOT NULL,
                content_hash VARCHAR,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                processing_status VARCHAR DEFAULT 'completed',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_file_metadata_status ON file_metadata(processing_status)",
            "CREATE INDEX IF NOT EXISTS idx_file_metadata_modified ON file_metadata(file_modified_at)",
            "CREATE INDEX IF NOT EXISTS idx_file_metadata_processed ON file_metadata(last_processed_at)",
        ],
        rollback_sql=[
            "DROP TABLE IF EXISTS file_metadata",
        ],
    ),
    # Future migrations can be added here
}
