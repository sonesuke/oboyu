"""Tests for file metadata schema and migrations.

This module tests the file_metadata table schema definition and
migration functionality.
"""

import pytest

from oboyu.indexer.storage.schema import SCHEMA_MIGRATIONS, DatabaseSchema


class TestFileMetadataSchema:
    """Test file metadata schema definition."""
    
    @pytest.fixture
    def schema(self):
        """Create a DatabaseSchema instance."""
        return DatabaseSchema()
    
    def test_file_metadata_table_definition(self, schema):
        """Test file_metadata table is properly defined."""
        # Get the file metadata table definition
        file_metadata_table = schema.get_file_metadata_table()
        
        # Verify table name
        assert file_metadata_table.name == "file_metadata"
        
        # Verify SQL contains all required columns
        sql = file_metadata_table.sql
        assert "path VARCHAR PRIMARY KEY" in sql
        assert "last_processed_at TIMESTAMP NOT NULL" in sql
        assert "file_modified_at TIMESTAMP NOT NULL" in sql
        assert "file_size BIGINT NOT NULL" in sql
        assert "content_hash VARCHAR" in sql
        assert "chunk_count INTEGER NOT NULL DEFAULT 0" in sql
        assert "processing_status VARCHAR DEFAULT 'completed'" in sql
        assert "error_message TEXT" in sql
        assert "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP" in sql
        assert "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP" in sql
        
        # Verify indexes
        assert len(file_metadata_table.indexes) == 3
        index_sqls = file_metadata_table.indexes
        assert any("idx_file_metadata_status" in idx for idx in index_sqls)
        assert any("idx_file_metadata_modified" in idx for idx in index_sqls)
        assert any("idx_file_metadata_processed" in idx for idx in index_sqls)
        
        # Verify no dependencies
        assert file_metadata_table.dependencies == []
    
    def test_file_metadata_table_in_all_tables(self, schema):
        """Test that file_metadata table is included in all tables list."""
        all_tables = schema.get_all_tables()
        
        # Find file_metadata table
        file_metadata_found = False
        for table in all_tables:
            if table.name == "file_metadata":
                file_metadata_found = True
                break
        
        assert file_metadata_found, "file_metadata table not found in all tables"
    
    def test_file_metadata_migration(self):
        """Test the file metadata migration definition."""
        # Check that migration 1.1.0 exists
        assert "1.1.0" in SCHEMA_MIGRATIONS
        
        migration = SCHEMA_MIGRATIONS["1.1.0"]
        
        # Verify migration metadata
        assert migration.version == "1.1.0"
        assert "file metadata" in migration.description.lower()
        
        # Verify migration SQL
        assert len(migration.migration_sql) >= 4  # Table + 3 indexes
        
        # Check table creation SQL
        create_table_sql = migration.migration_sql[0]
        assert "CREATE TABLE IF NOT EXISTS file_metadata" in create_table_sql
        assert "path VARCHAR PRIMARY KEY" in create_table_sql
        assert "last_processed_at TIMESTAMP NOT NULL" in create_table_sql
        assert "file_modified_at TIMESTAMP NOT NULL" in create_table_sql
        assert "file_size BIGINT NOT NULL" in create_table_sql
        assert "content_hash VARCHAR" in create_table_sql
        assert "chunk_count INTEGER NOT NULL DEFAULT 0" in create_table_sql
        assert "processing_status VARCHAR DEFAULT 'completed'" in create_table_sql
        assert "error_message TEXT" in create_table_sql
        
        # Check index creation SQLs
        index_sqls = migration.migration_sql[1:4]
        assert any("CREATE INDEX IF NOT EXISTS idx_file_metadata_status" in sql for sql in index_sqls)
        assert any("CREATE INDEX IF NOT EXISTS idx_file_metadata_modified" in sql for sql in index_sqls)
        assert any("CREATE INDEX IF NOT EXISTS idx_file_metadata_processed" in sql for sql in index_sqls)
        
        # Verify rollback SQL
        assert len(migration.rollback_sql) == 1
        assert migration.rollback_sql[0] == "DROP TABLE IF EXISTS file_metadata"
    
    def test_schema_consistency_with_migration(self, schema):
        """Test that schema definition matches migration SQL."""
        # Get table definition from schema
        file_metadata_table = schema.get_file_metadata_table()
        
        # Get migration SQL
        migration = SCHEMA_MIGRATIONS["1.1.0"]
        migration_table_sql = migration.migration_sql[0]
        
        # Both should have the same columns (though formatting might differ)
        schema_columns = [
            "path VARCHAR PRIMARY KEY",
            "last_processed_at TIMESTAMP NOT NULL",
            "file_modified_at TIMESTAMP NOT NULL", 
            "file_size BIGINT NOT NULL",
            "content_hash VARCHAR",
            "chunk_count INTEGER NOT NULL DEFAULT 0",
            "processing_status VARCHAR DEFAULT 'completed'",
            "error_message TEXT",
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ]
        
        # Check that all columns are present in both definitions
        for column in schema_columns:
            assert column in file_metadata_table.sql, f"Column '{column}' missing from schema definition"
            assert column in migration_table_sql, f"Column '{column}' missing from migration SQL"
    
    def test_drop_tables_includes_file_metadata(self, schema):
        """Test that drop all tables includes file_metadata."""
        drop_sqls = schema.get_drop_all_tables_sql()
        
        # Check that file_metadata is included
        file_metadata_drop_found = False
        for sql in drop_sqls:
            if "DROP TABLE IF EXISTS file_metadata" in sql:
                file_metadata_drop_found = True
                break
        
        assert file_metadata_drop_found, "file_metadata not included in drop tables"
    
    def test_schema_validation_passes(self, schema):
        """Test that schema validation passes with file_metadata table."""
        # This should not raise any validation errors
        issues = schema.validate_schema_consistency()
        
        # Should have no issues
        assert len(issues) == 0, f"Schema validation failed: {issues}"