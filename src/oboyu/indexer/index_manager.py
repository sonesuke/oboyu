"""Index management for Oboyu indexer database.

This module provides centralized management of database indexes including
HNSW vector indexes and BM25 search indexes. It handles index creation,
validation, optimization, and health monitoring.

Key features:
- Centralized HNSW index management with parameter validation
- BM25 index optimization and maintenance
- Index health monitoring and diagnostics
- Safe index recreation and recovery
- Performance optimization strategies
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from duckdb import DuckDBPyConnection

from oboyu.indexer.schema import DatabaseSchema

logger = logging.getLogger(__name__)


@dataclass
class IndexInfo:
    """Information about a database index."""
    
    name: str
    table: str
    columns: List[str]
    index_type: str
    is_unique: bool
    is_valid: bool
    size_estimate: Optional[int] = None


@dataclass
class HNSWIndexParams:
    """Parameters for HNSW index configuration."""
    
    ef_construction: int = 128
    ef_search: int = 64
    m: int = 16
    m0: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate and set default values."""
        if self.m0 is None:
            self.m0 = 2 * self.m
        
        # Validate parameter ranges
        if not (10 <= self.ef_construction <= 2000):
            raise ValueError("ef_construction must be between 10 and 2000")
        if not (1 <= self.ef_search <= 1000):
            raise ValueError("ef_search must be between 1 and 1000")
        if not (2 <= self.m <= 100):
            raise ValueError("m must be between 2 and 100")
        if not (2 <= self.m0 <= 200):
            raise ValueError("m0 must be between 2 and 200")


@dataclass
class IndexHealth:
    """Health status of database indexes."""
    
    total_indexes: int
    healthy_indexes: int
    broken_indexes: List[str]
    warnings: List[str]
    recommendations: List[str]


class IndexManager:
    """Centralized database index management.
    
    This class provides methods for creating, maintaining, and optimizing
    all database indexes including HNSW vector indexes and BM25 search indexes.
    It ensures indexes are properly configured and maintained for optimal performance.
    """
    
    def __init__(self, conn: DuckDBPyConnection, schema: DatabaseSchema) -> None:
        """Initialize index manager.
        
        Args:
            conn: Database connection
            schema: Database schema manager

        """
        self.conn = conn
        self.schema = schema
        self._hnsw_params: Optional[HNSWIndexParams] = None
    
    def setup_all_indexes(self, hnsw_params: Optional[HNSWIndexParams] = None) -> None:
        """Set up all database indexes.
        
        Args:
            hnsw_params: Optional HNSW index parameters

        """
        logger.info("Setting up database indexes...")
        
        # Store HNSW parameters for later use
        if hnsw_params:
            self._hnsw_params = hnsw_params
        
        # Create standard table indexes
        self._create_table_indexes()
        
        # Create HNSW index if embeddings table has data
        if self._should_create_hnsw_index():
            self.create_hnsw_index(hnsw_params or HNSWIndexParams())
        
        logger.info("Database indexes setup completed")
    
    def _create_table_indexes(self) -> None:
        """Create all standard table indexes."""
        tables = self.schema.get_all_tables()
        
        for table in tables:
            for index_sql in table.indexes:
                try:
                    self.conn.execute(index_sql)
                    logger.debug(f"Created index: {index_sql.split()[:3]}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
    
    def create_hnsw_index(self, params: HNSWIndexParams, force: bool = False) -> bool:
        """Create HNSW vector similarity index.
        
        Args:
            params: HNSW index parameters
            force: Whether to recreate if index exists
            
        Returns:
            True if index was created successfully

        """
        try:
            # Check if index already exists
            if not force and self.hnsw_index_exists():
                logger.info("HNSW index already exists - skipping creation")
                return True
            
            # Drop existing index if forcing recreation
            if force:
                self.drop_hnsw_index()
            
            # Create the index
            index_sql = self.schema.get_hnsw_index_sql(
                ef_construction=params.ef_construction,
                ef_search=params.ef_search,
                m=params.m,
                m0=params.m0
            )
            
            logger.info(f"Creating HNSW index with parameters: ef_construction={params.ef_construction}, "
                       f"ef_search={params.ef_search}, m={params.m}, m0={params.m0}")
            
            self.conn.execute(index_sql)
            self._hnsw_params = params
            
            logger.info("HNSW index created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create HNSW index: {e}")
            return False
    
    def drop_hnsw_index(self) -> bool:
        """Drop the HNSW vector index.
        
        Returns:
            True if index was dropped successfully

        """
        try:
            self.conn.execute("DROP INDEX IF EXISTS vector_idx")
            logger.info("HNSW index dropped")
            return True
        except Exception as e:
            logger.error(f"Failed to drop HNSW index: {e}")
            return False
    
    def hnsw_index_exists(self) -> bool:
        """Check if HNSW index exists.
        
        Returns:
            True if HNSW index exists

        """
        try:
            result = self.conn.execute("""
                SELECT COUNT(*)
                FROM duckdb_indexes
                WHERE index_name = 'vector_idx'
            """).fetchone()
            return bool(result and result[0] > 0)
        except Exception:
            return False
    
    def _should_create_hnsw_index(self) -> bool:
        """Check if HNSW index should be created.
        
        Returns:
            True if embeddings table has data

        """
        try:
            result = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            return bool(result and result[0] > 0)
        except Exception:
            return False
    
    def recreate_hnsw_index(self, params: Optional[HNSWIndexParams] = None) -> bool:
        """Recreate the HNSW index.
        
        Args:
            params: Optional new parameters (uses existing if not provided)
            
        Returns:
            True if recreation was successful

        """
        # Use existing parameters if not provided
        if params is None:
            params = self._hnsw_params or HNSWIndexParams()
        
        logger.info("Recreating HNSW index...")
        return self.create_hnsw_index(params, force=True)
    
    def validate_hnsw_index_params(self, expected_params: HNSWIndexParams) -> bool:
        """Validate that existing HNSW index has expected parameters.
        
        Args:
            expected_params: Expected index parameters
            
        Returns:
            True if parameters match or validation is not possible
            
        Note: DuckDB doesn't currently expose index parameters for validation.
        This method is a placeholder for future functionality.

        """
        # TODO: Implement parameter validation when DuckDB exposes index metadata
        # For now, assume index is valid if it exists
        if not self.hnsw_index_exists():
            return False
        
        # Store the expected parameters for future use
        self._hnsw_params = expected_params
        return True
    
    def compact_hnsw_index(self) -> bool:
        """Compact the HNSW index to improve search performance.
        
        Returns:
            True if compaction was successful

        """
        try:
            if not self.hnsw_index_exists():
                logger.warning("Cannot compact HNSW index - index does not exist")
                return False
            
            self.conn.execute("PRAGMA hnsw_compact_index('vector_idx')")
            logger.info("HNSW index compacted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to compact HNSW index: {e}")
            return False
    
    def optimize_bm25_indexes(self) -> bool:
        """Optimize BM25 search indexes.
        
        Returns:
            True if optimization was successful

        """
        try:
            # Analyze tables to update statistics
            bm25_tables = ["vocabulary", "inverted_index", "document_stats", "collection_stats"]
            
            for table in bm25_tables:
                try:
                    self.conn.execute(f"ANALYZE {table}")
                    logger.debug(f"Analyzed table: {table}")
                except Exception as e:
                    logger.warning(f"Failed to analyze table {table}: {e}")
            
            logger.info("BM25 indexes optimized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize BM25 indexes: {e}")
            return False
    
    def get_index_info(self) -> List[IndexInfo]:
        """Get information about all database indexes.
        
        Returns:
            List of index information

        """
        try:
            # Query DuckDB system tables for index information
            result = self.conn.execute("""
                SELECT
                    index_name,
                    table_name,
                    sql,
                    is_unique
                FROM duckdb_indexes
                WHERE NOT index_name LIKE 'pg_%'  -- Exclude system indexes
                ORDER BY table_name, index_name
            """).fetchall()
            
            indexes = []
            for row in result:
                index_name = row[0]
                table_name = row[1]
                sql = row[2] or ""
                is_unique = bool(row[3])
                
                # Parse index type from SQL
                index_type = "btree"  # Default
                if "HNSW" in sql.upper():
                    index_type = "hnsw"
                elif "GIN" in sql.upper():
                    index_type = "gin"
                
                # Extract columns (simplified parsing)
                columns = [index_name.replace("idx_", "").replace(f"{table_name}_", "")]
                
                indexes.append(IndexInfo(
                    name=index_name,
                    table=table_name,
                    columns=columns,
                    index_type=index_type,
                    is_unique=is_unique,
                    is_valid=True  # Assume valid since DuckDB doesn't track broken indexes
                ))
            
            return indexes
            
        except Exception as e:
            logger.error(f"Failed to get index information: {e}")
            return []
    
    def check_index_health(self) -> IndexHealth:
        """Check the health of all database indexes.
        
        Returns:
            Index health status and recommendations

        """
        indexes = self.get_index_info()
        warnings: List[str] = []
        recommendations: List[str] = []
        broken_indexes: List[str] = []
        
        # Check HNSW index status
        if self._should_create_hnsw_index() and not self.hnsw_index_exists():
            warnings.append("HNSW index missing despite having embedding data")
            recommendations.append("Create HNSW index for vector search performance")
        
        # Check for missing indexes on large tables
        try:
            # Check chunks table size
            chunk_count_result = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            chunk_count = chunk_count_result[0] if chunk_count_result else 0
            
            if chunk_count > 10000:
                chunk_indexes = [idx for idx in indexes if idx.table == "chunks"]
                if len(chunk_indexes) < 2:  # Should have path and language indexes
                    warnings.append(f"Large chunks table ({chunk_count} rows) has few indexes")
                    recommendations.append("Consider adding indexes on frequently queried columns")
            
            # Check inverted index table size
            ii_count_result = self.conn.execute("SELECT COUNT(*) FROM inverted_index").fetchone()
            ii_count = ii_count_result[0] if ii_count_result else 0
            
            if ii_count > 100000:
                ii_indexes = [idx for idx in indexes if idx.table == "inverted_index"]
                if len(ii_indexes) < 2:  # Should have term and chunk_id indexes
                    warnings.append(f"Large inverted_index table ({ii_count} rows) may need more indexes")
                    recommendations.append("Ensure BM25 search indexes are properly created")
                    
        except Exception as e:
            warnings.append(f"Failed to check table sizes: {e}")
        
        # Performance recommendations
        if len(indexes) > 20:
            recommendations.append("Consider removing unused indexes to improve write performance")
        
        return IndexHealth(
            total_indexes=len(indexes),
            healthy_indexes=len(indexes) - len(broken_indexes),
            broken_indexes=broken_indexes,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def rebuild_all_indexes(self) -> bool:
        """Rebuild all database indexes.
        
        Returns:
            True if all indexes were rebuilt successfully

        """
        logger.info("Rebuilding all database indexes...")
        
        try:
            # Drop and recreate HNSW index
            if self.hnsw_index_exists():
                hnsw_success = self.recreate_hnsw_index()
                if not hnsw_success:
                    logger.error("Failed to rebuild HNSW index")
                    return False
            
            # Rebuild standard indexes
            self._create_table_indexes()
            
            # Optimize BM25 indexes
            self.optimize_bm25_indexes()
            
            logger.info("All indexes rebuilt successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild indexes: {e}")
            return False
    
    def get_index_usage_stats(self) -> Dict[str, Any]:
        """Get index usage statistics.
        
        Returns:
            Dictionary with index usage information
            
        Note: DuckDB doesn't provide detailed index usage statistics.
        This method provides basic information available.

        """
        stats: Dict[str, Any] = {
            "total_indexes": 0,
            "hnsw_index_exists": self.hnsw_index_exists(),
            "hnsw_params": self._hnsw_params.__dict__ if self._hnsw_params else None,
            "indexes": []
        }
        
        try:
            indexes = self.get_index_info()
            stats["total_indexes"] = len(indexes)
            stats["indexes"] = [
                {
                    "name": idx.name,
                    "table": idx.table,
                    "type": idx.index_type,
                    "unique": idx.is_unique
                }
                for idx in indexes
            ]
            
        except Exception as e:
            logger.error(f"Failed to get index usage stats: {e}")
        
        return stats
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance recommendations for index optimization.
        
        Returns:
            List of recommendation strings

        """
        recommendations = []
        health = self.check_index_health()
        
        # Add health-based recommendations
        recommendations.extend(health.recommendations)
        
        # Add general performance recommendations
        try:
            # Check if we should compact HNSW index
            if self.hnsw_index_exists():
                embedding_count_result = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
                embedding_count = embedding_count_result[0] if embedding_count_result else 0
                
                if embedding_count > 50000:
                    recommendations.append("Consider compacting HNSW index for large embedding collections")
            
            # Check for table maintenance
            chunk_count_result = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            chunk_count = chunk_count_result[0] if chunk_count_result else 0
            
            if chunk_count > 100000:
                recommendations.append("Consider running VACUUM or ANALYZE for large tables")
                
        except Exception as e:
            logger.error(f"Failed to generate performance recommendations: {e}")
        
        return recommendations
