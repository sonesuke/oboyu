"""HNSW vector index management for DuckDB VSS extension."""

import logging
from dataclasses import dataclass
from typing import Optional

from duckdb import DuckDBPyConnection

from .schema import DatabaseSchema

logger = logging.getLogger(__name__)


@dataclass
class HNSWIndexParams:
    """Parameters for HNSW index configuration."""
    
    ef_construction: int = 128
    ef_search: int = 64
    m: int = 16
    m0: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.ef_construction < 1:
            raise ValueError("ef_construction must be >= 1")
        if self.ef_search < 1:
            raise ValueError("ef_search must be >= 1")
        if self.m < 1:
            raise ValueError("m must be >= 1")
        if self.m0 is not None and self.m0 < 1:
            raise ValueError("m0 must be >= 1 if provided")


class HNSWIndexManager:
    """Manages HNSW vector similarity indexes in DuckDB."""

    def __init__(self, conn: DuckDBPyConnection, schema: DatabaseSchema) -> None:
        """Initialize HNSW index manager.
        
        Args:
            conn: Database connection
            schema: Database schema definition

        """
        self.conn = conn
        self.schema = schema
        self._hnsw_params: Optional[HNSWIndexParams] = None

    def create_hnsw_index(self, params: HNSWIndexParams, force: bool = False) -> bool:
        """Create HNSW vector similarity index with concurrent access handling.

        Args:
            params: HNSW index parameters
            force: Whether to recreate if index exists

        Returns:
            True if index was created successfully

        """
        import time
        
        max_retries = 3
        retry_delay = 0.5  # 500ms initial delay
        
        for attempt in range(max_retries):
            try:
                # Check if index already exists
                if not force and self.hnsw_index_exists():
                    logger.debug("HNSW index already exists - skipping creation")
                    return True

                # Drop existing index if forcing recreation
                if force:
                    self.drop_hnsw_index()

                # Check if embeddings table has data
                count_result = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
                count = count_result[0] if count_result else 0
                
                if count == 0:
                    logger.info("No embeddings found, skipping HNSW index creation")
                    return True

                # Create the index
                index_sql = self.schema.get_hnsw_index_sql(
                    ef_construction=params.ef_construction,
                    ef_search=params.ef_search,
                    m=params.m,
                    m0=params.m0
                )

                logger.info(
                    f"Creating HNSW index with parameters: ef_construction={params.ef_construction}, "
                    f"ef_search={params.ef_search}, m={params.m}, m0={params.m0} (attempt {attempt + 1}/{max_retries})"
                )

                self.conn.execute(index_sql)
                self._hnsw_params = params

                logger.info("HNSW index created successfully")
                return True

            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a concurrent access error
                if ("lock" in error_msg or "concurrent" in error_msg or "already exists" in error_msg):
                    if attempt < max_retries - 1:
                        logger.warning(f"HNSW index creation failed due to concurrent access, retrying in {retry_delay}s: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        
                        # Check again if index was created by another process
                        if self.hnsw_index_exists():
                            logger.info("HNSW index was created by another process")
                            return True
                        continue
                    
                logger.error(f"Failed to create HNSW index after {attempt + 1} attempts: {e}")
                if 'index_sql' in locals():
                    logger.debug(f"Index SQL was: {index_sql}")
                
                return False
                
        return False

    def drop_hnsw_index(self) -> bool:
        """Drop the HNSW vector index.
        
        Returns:
            True if index was dropped successfully

        """
        try:
            self.conn.execute("DROP INDEX IF EXISTS vector_idx")
            logger.info("HNSW index dropped successfully")
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
            result = self.conn.execute(
                "SELECT COUNT(*) FROM duckdb_indexes() WHERE index_name = 'vector_idx'"
            ).fetchone()
            return bool(result and result[0] > 0)
        except Exception:
            return False

    def should_create_hnsw_index(self) -> bool:
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
        """Validate current HNSW index parameters against expected values.

        Args:
            expected_params: Expected parameter values

        Returns:
            True if parameters match expectations

        """
        if not self._hnsw_params:
            logger.warning("No HNSW parameters stored - validation skipped")
            return False

        current = self._hnsw_params
        
        # Check parameter consistency
        params_match = (
            current.ef_construction == expected_params.ef_construction and
            current.ef_search == expected_params.ef_search and
            current.m == expected_params.m and
            current.m0 == expected_params.m0
        )

        if not params_match:
            logger.warning(f"HNSW parameter mismatch: current={current}, expected={expected_params}")

        return params_match

    def compact_hnsw_index(self) -> bool:
        """Compact the HNSW index to optimize storage and performance.
        
        Returns:
            True if compaction was successful

        """
        try:
            if not self.hnsw_index_exists():
                logger.warning("Cannot compact HNSW index - index does not exist")
                return False

            # For now, we recreate the index as DuckDB doesn't have explicit compaction
            logger.info("Compacting HNSW index by recreation...")
            return self.recreate_hnsw_index()

        except Exception as e:
            logger.error(f"Failed to compact HNSW index: {e}")
            return False

    def get_hnsw_params(self) -> Optional[HNSWIndexParams]:
        """Get current HNSW index parameters.
        
        Returns:
            Current HNSW parameters or None if not set

        """
        return self._hnsw_params
