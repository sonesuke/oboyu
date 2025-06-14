"""Knowledge Graph schema definitions for Oboyu indexer.

This module provides database schema definitions specifically for the Property Graph Index
functionality, including entities, relations, and processing status tracking.

Key features:
- Entity and relation extraction from existing chunks
- Property graph schema compatible with DuckPGQ
- Delta update tracking via processing status
- Entity deduplication support with canonical names
- Confidence scoring for extraction quality
"""

from dataclasses import dataclass
from typing import List

from .schema_types import TableDefinition


@dataclass
class KnowledgeGraphSchema:
    """Knowledge Graph specific database schema definitions.

    This class provides table definitions for the Property Graph Index functionality,
    designed to work alongside the existing oboyu core schema.
    """

    def get_kg_entities_table(self) -> TableDefinition:
        """Get knowledge graph entities table definition."""
        return TableDefinition(
            name="kg_entities",
            sql="""
                CREATE TABLE IF NOT EXISTS kg_entities (
                    id VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL,               -- Entity name/label
                    entity_type VARCHAR NOT NULL,        -- PERSON, COMPANY, PRODUCT, etc.
                    definition TEXT,                     -- LLM-generated entity definition
                    properties JSON,                     -- Additional properties and metadata
                    chunk_id VARCHAR,                    -- Source chunk ID
                    canonical_name VARCHAR,              -- Canonical name for deduplication
                    merged_from JSON,                    -- Array of entity IDs merged into this one
                    merge_confidence REAL,               -- Confidence score for entity merging
                    confidence REAL NOT NULL,            -- Extraction confidence score
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
                )
            """.strip(),
            indexes=[
                "CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities(name)",
                "CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type)",
                "CREATE INDEX IF NOT EXISTS idx_kg_entities_chunk ON kg_entities(chunk_id)",
                "CREATE INDEX IF NOT EXISTS idx_kg_entities_canonical ON kg_entities(canonical_name)",
                "CREATE INDEX IF NOT EXISTS idx_kg_entities_confidence ON kg_entities(confidence)",
            ],
            dependencies=["chunks"],
        )

    def get_kg_relations_table(self) -> TableDefinition:
        """Get knowledge graph relations table definition."""
        return TableDefinition(
            name="kg_relations",
            sql="""
                CREATE TABLE IF NOT EXISTS kg_relations (
                    id VARCHAR PRIMARY KEY,
                    source_id VARCHAR NOT NULL,          -- Source entity ID
                    target_id VARCHAR NOT NULL,          -- Target entity ID
                    relation_type VARCHAR NOT NULL,      -- WORKS_AT, CEO_OF, LOCATED_IN, etc.
                    properties JSON,                     -- Relation metadata and temporal info
                    chunk_id VARCHAR,                    -- Source chunk ID
                    confidence REAL NOT NULL,            -- Extraction confidence score
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES kg_entities (id),
                    FOREIGN KEY (target_id) REFERENCES kg_entities (id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
                )
            """.strip(),
            indexes=[
                "CREATE INDEX IF NOT EXISTS idx_kg_relations_source ON kg_relations(source_id)",
                "CREATE INDEX IF NOT EXISTS idx_kg_relations_target ON kg_relations(target_id)",
                "CREATE INDEX IF NOT EXISTS idx_kg_relations_type ON kg_relations(relation_type)",
                "CREATE INDEX IF NOT EXISTS idx_kg_relations_chunk ON kg_relations(chunk_id)",
                "CREATE INDEX IF NOT EXISTS idx_kg_relations_confidence ON kg_relations(confidence)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_relations_unique ON kg_relations(source_id, target_id, relation_type)",
            ],
            dependencies=["kg_entities", "chunks"],
        )

    def get_kg_processing_status_table(self) -> TableDefinition:
        """Get knowledge graph processing status table for delta updates."""
        return TableDefinition(
            name="kg_processing_status",
            sql="""
                CREATE TABLE IF NOT EXISTS kg_processing_status (
                    chunk_id VARCHAR PRIMARY KEY,
                    processed_at TIMESTAMP NOT NULL,     -- When KG extraction was performed
                    processing_version VARCHAR NOT NULL, -- Version of processing pipeline
                    entity_count INTEGER DEFAULT 0,      -- Number of entities extracted
                    relation_count INTEGER DEFAULT 0,    -- Number of relations extracted
                    processing_time_ms INTEGER,          -- Processing time in milliseconds
                    model_used VARCHAR,                   -- LLM model used for extraction
                    error_message TEXT,                   -- Error message if processing failed
                    status VARCHAR DEFAULT 'completed',  -- 'completed', 'error', 'in_progress'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES chunks (id)
                )
            """.strip(),
            indexes=[
                "CREATE INDEX IF NOT EXISTS idx_kg_processing_status_processed ON kg_processing_status(processed_at)",
                "CREATE INDEX IF NOT EXISTS idx_kg_processing_status_version ON kg_processing_status(processing_version)",
                "CREATE INDEX IF NOT EXISTS idx_kg_processing_status_status ON kg_processing_status(status)",
            ],
            dependencies=["chunks"],
        )

    def get_all_kg_tables(self) -> List[TableDefinition]:
        """Get all knowledge graph table definitions.

        Returns:
            List of KG table definitions in dependency order

        """
        return [
            self.get_kg_entities_table(),
            self.get_kg_relations_table(),
            self.get_kg_processing_status_table(),
        ]

    def get_duckpgq_property_graph_sql(self) -> str:
        """Get DuckPGQ property graph creation SQL.

        Returns:
            SQL for creating the property graph structure

        """
        return """
            CREATE PROPERTY GRAPH oboyu_kg
            VERTEX TABLES (kg_entities LABEL entity)
            EDGE TABLES (
                kg_relations
                SOURCE KEY (source_id) REFERENCES kg_entities (id)
                DESTINATION KEY (target_id) REFERENCES kg_entities (id)
                LABEL relation
            )
        """.strip()
