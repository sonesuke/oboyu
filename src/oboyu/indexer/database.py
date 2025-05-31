"""Database management for Oboyu indexer.

This module handles the DuckDB database operations including schema setup,
VSS (Vector Similarity Search) extension, HNSW index management, and batched
document processing.

The implementation provides:
- Efficient processing of large document collections
- Optimized batch sizes for performance and memory usage
- Automatic batching for large datasets to control memory usage
- Support for vector similarity search using HNSW index

Key components:
- DuckDB with VSS extension for vector similarity search
- HNSW index for efficient vector search
- Configurable batch sizes with validation
- Automatic batching for large datasets to control memory usage
"""

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from oboyu.indexer.config import DEFAULT_BATCH_SIZE
from oboyu.indexer.database_manager import DatabaseManager
from oboyu.indexer.index_manager import HNSWIndexParams
from oboyu.indexer.processor import Chunk
from oboyu.indexer.queries import ChunkData, EmbeddingData, QueryBuilder

logger = logging.getLogger(__name__)


# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""

    def default(self, obj: object) -> object:
        """Convert datetime objects to ISO format strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)



class Database:
    """Manager for the vector database using DuckDB with VSS extension.

    This class provides functionality for storing, retrieving, and searching vector data
    using DuckDB's VSS extension for vector similarity search. The implementation
    optimizes for both performance and memory efficiency when handling large document
    collections through batched operations.

    Key features:
    - Batched processing for efficient document and embedding handling
    - Configurable batch sizes for optimization
    - Automatic batching for large datasets
    - Vector similarity search using HNSW index
    - Support for document metadata and language filtering

    Future versions will integrate ADBC for higher performance batch operations
    once compatibility issues with the VSS extension are resolved.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        embedding_dimensions: int = 256,
        ef_construction: int = 128,
        ef_search: int = 64,
        m: int = 16,
        m0: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialize the database.

        Args:
            db_path: Path to the database file (required, no default)
            embedding_dimensions: Dimensions of the embedding vectors
            ef_construction: Index construction parameter (build-time)
            ef_search: Search time parameter (quality vs. speed)
            m: Number of bidirectional links in HNSW graph
            m0: Level-0 connections (None means use 2*M)
            batch_size: Batch size for ADBC operations (defaults to optimal batch size)

        """
        self.db_path = Path(db_path)
        self.embedding_dimensions = embedding_dimensions
        self.batch_size = batch_size
        
        # Create HNSW parameters
        self.hnsw_params = HNSWIndexParams(
            ef_construction=ef_construction,
            ef_search=ef_search,
            m=m,
            m0=m0 if m0 is not None else 2 * m
        )
        
        # Initialize new database manager
        self.db_manager = DatabaseManager(
            db_path=db_path,
            embedding_dimensions=embedding_dimensions,
            hnsw_params=self.hnsw_params
        )
        
        # Backward compatibility properties
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m = m
        self.m0 = m0 if m0 is not None else 2 * m
    
    @property
    def conn(self) -> Optional[DuckDBPyConnection]:
        """Get database connection for backward compatibility."""
        if hasattr(self, 'db_manager') and self.db_manager:
            return self.db_manager.connection
        return getattr(self, '_conn', None)
    
    @conn.setter  
    def conn(self, value: Optional[DuckDBPyConnection]) -> None:
        """Set database connection (for backward compatibility)."""
        # Store for backward compatibility, but prefer using db_manager
        self._conn = value


    def setup(self) -> None:
        """Set up the database schema and extensions."""
        # Use the new database manager for setup
        self.db_manager.initialize_database()



    def _create_schema(self) -> None:
        """Create database schema."""
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # Create chunks table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id VARCHAR PRIMARY KEY,
                path VARCHAR,             -- Path to file
                title VARCHAR,            -- Chunk title (or original filename)
                content TEXT,             -- Chunk text content
                chunk_index INTEGER,      -- Chunk position in original document
                language VARCHAR,         -- Language code
                created_at TIMESTAMP,     -- Creation timestamp
                modified_at TIMESTAMP,    -- Modification timestamp
                metadata JSON             -- Additional metadata
            )
        """)

        # Create embeddings table
        # DuckDB doesn't support ? in DDL statements, use string formatting carefully
        # This is controlled input from self.embedding_dimensions, not user input
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id VARCHAR PRIMARY KEY,
                chunk_id VARCHAR,         -- Related chunk ID
                model VARCHAR,            -- Embedding model used
                vector FLOAT[{self.embedding_dimensions}],  -- Vector dimensions specific to model
                created_at TIMESTAMP,     -- Embedding generation timestamp
                FOREIGN KEY (chunk_id) REFERENCES chunks (id)
            )
        """)

        # Check if HNSW index exists and is valid
        if not self._hnsw_index_exists():
            logger.info("Creating new HNSW index...")
            self._create_hnsw_index()
        else:
            logger.info("Using existing HNSW index - skipping index creation for faster startup")
            # Optionally validate index parameters
            if not self._validate_hnsw_index_params():
                logger.warning("HNSW index parameters mismatch, recreating index...")
                self.conn.execute("DROP INDEX IF EXISTS vector_idx")
                self._create_hnsw_index()
        
        # Create BM25 schema
        self._create_bm25_schema()
    
    def _create_bm25_schema(self) -> None:
        """Create BM25-related database schema."""
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")
        
        # Create vocabulary table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vocabulary (
                term VARCHAR PRIMARY KEY,
                document_frequency INTEGER,
                collection_frequency INTEGER
            )
        """)
        
        # Create inverted index table without primary key for faster bulk inserts
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS inverted_index (
                term VARCHAR,
                chunk_id VARCHAR,
                term_frequency INTEGER,
                positions INTEGER[]  -- Array of token positions for phrase search
            )
        """)
        
        # Create document statistics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS document_stats (
                chunk_id VARCHAR PRIMARY KEY,
                total_terms INTEGER,
                unique_terms INTEGER,
                avg_term_frequency REAL,
                FOREIGN KEY (chunk_id) REFERENCES chunks (id)
            )
        """)
        
        # Create collection statistics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS collection_stats (
                id INTEGER PRIMARY KEY DEFAULT 1,  -- Single row
                total_documents INTEGER,
                total_terms INTEGER,
                avg_document_length REAL,
                last_updated TIMESTAMP
            )
        """)
        
        # Create indexes for BM25 search performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_inverted_index_term ON inverted_index(term)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_inverted_index_chunk ON inverted_index(chunk_id)")

    def store_chunks(self, chunks: List[Chunk]) -> None:
        """Store document chunks in the database using ADBC for efficient batch operations.

        Args:
            chunks: List of document chunks to store

        """
        if not chunks:
            return

        # Convert chunks to ChunkData for type safety
        chunk_data_list = [QueryBuilder.from_chunk_to_chunk_data(chunk) for chunk in chunks]

        # Use transaction for batch operations
        with self.db_manager.transaction() as conn:
            # Process chunks in batches
            for i in range(0, len(chunk_data_list), self.batch_size):
                batch = chunk_data_list[i:i + self.batch_size]
                
                # Use QueryBuilder for type-safe operations
                for chunk_data in batch:
                    sql, params = QueryBuilder.upsert_chunk(chunk_data)
                    conn.execute(sql, params)


    def store_embeddings(
        self,
        embeddings: List[Tuple[str, str, NDArray[np.float32], datetime]],
        model_name: str,
    ) -> None:
        """Store embeddings in the database using ADBC for efficient batch operations.

        Args:
            embeddings: List of (id, chunk_id, embedding, timestamp) tuples
            model_name: Name of the embedding model used

        """
        if not embeddings:
            return

        # Convert to EmbeddingData for type safety
        embedding_data_list = []
        for embedding_id, chunk_id, vector, timestamp in embeddings:
            # Skip problematic embeddings with scalar shape
            if vector.shape == ():
                continue
            
            embedding_data_list.append(EmbeddingData(
                id=embedding_id,
                chunk_id=chunk_id,
                model=model_name,
                vector=vector,
                created_at=timestamp
            ))

        # Use transaction for batch operations
        with self.db_manager.transaction() as conn:
            # Process embeddings in batches to control memory usage
            for i in range(0, len(embedding_data_list), self.batch_size):
                batch = embedding_data_list[i:i + self.batch_size]
                
                # Use QueryBuilder for type-safe operations
                for embedding_data in batch:
                    sql, params = QueryBuilder.upsert_embedding(embedding_data)
                    conn.execute(sql, params)

    def _store_chunks_duckdb(self, chunks: List[Chunk]) -> None:
        """Store document chunks using standard DuckDB (fallback method).

        Args:
            chunks: List of document chunks to store

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")
        
        # Store each chunk individually using standard DuckDB
        batch_data = []
        for chunk in chunks:
            # Convert metadata to JSON string
            metadata_json = json.dumps(chunk.metadata, cls=DateTimeEncoder) if chunk.metadata else None

            batch_data.append((
                chunk.id,
                str(chunk.path),
                chunk.title,
                chunk.content,
                chunk.chunk_index,
                chunk.language,
                chunk.created_at,
                chunk.modified_at,
                metadata_json
            ))

        # UPSERT operation
        self.conn.executemany("""
            INSERT INTO chunks (id, path, title, content, chunk_index, language, created_at, modified_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                path = excluded.path,
                title = excluded.title,
                content = excluded.content,
                chunk_index = excluded.chunk_index,
                language = excluded.language,
                modified_at = excluded.modified_at,
                metadata = excluded.metadata
        """, batch_data)
        
        self.conn.commit()


    def _store_embeddings_duckdb(
        self,
        embeddings: List[Tuple[str, str, NDArray[np.float32], datetime]],
        model_name: str,
    ) -> None:
        """Store embeddings using standard DuckDB (fallback method).

        Args:
            embeddings: List of embedding tuples (id, chunk_id, vector, timestamp)
            model_name: Name of the embedding model used

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")
        
        # Direct batch insert
        
        # Prepare batch data
        batch_data = []
        for embedding_id, chunk_id, vector, timestamp in embeddings:
            # Skip problematic embeddings with scalar shape
            if vector.shape == ():
                continue
            
            # Convert vector to list
            vector_list = vector.tolist()
            if not isinstance(vector_list, list):
                raise ValueError(f"Expected list from vector.tolist() but got {type(vector_list)}. Vector shape: {vector.shape}")
            
            batch_data.append((embedding_id, chunk_id, model_name, vector_list, timestamp))
        
        # Bulk insert directly
        self.conn.executemany("""
            INSERT INTO embeddings (id, chunk_id, model, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                chunk_id = excluded.chunk_id,
                model = excluded.model,
                vector = excluded.vector,
                created_at = excluded.created_at
        """, batch_data)
        
        self.conn.commit()

    def search(
        self,
        query_vector: NDArray[np.float32],
        limit: int = 10,
        language: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Search for similar documents using vector similarity.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            language: Filter by language (optional)

        Returns:
            List of matching document chunks with similarity scores

        """
        # Use QueryBuilder for type-safe vector search
        sql, params = QueryBuilder.search_by_vector(
            query_vector=query_vector,
            limit=limit,
            language=language,
            embedding_dimensions=self.embedding_dimensions
        )

        # Execute search using connection
        conn = self.db_manager.connection
        results = conn.execute(sql, params).fetchall()

        # Format results using QueryBuilder helper
        formatted_results = []
        for row in results:
            result_dict = QueryBuilder.search_result_from_row(row)
            
            # Convert cosine distance to similarity score for compatibility
            distance = result_dict["score"]
            similarity_score = 1.0 - (distance / 2.0)
            result_dict["score"] = similarity_score
            
            formatted_results.append(result_dict)

        return formatted_results

    def recompact_index(self) -> None:
        """Recompact the HNSW index to improve search performance."""
        # Use IndexManager for index operations
        self.db_manager.index_manager.compact_hnsw_index()

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, object]]:
        """Retrieve a chunk by its ID.

        Args:
            chunk_id: ID of the chunk to retrieve

        Returns:
            Chunk data as dictionary or None if not found

        """
        # Use QueryBuilder for type-safe query
        sql, params = QueryBuilder.select_chunk_by_id(chunk_id)
        
        conn = self.db_manager.connection
        result = conn.execute(sql, params).fetchone()

        if not result:
            return None

        return QueryBuilder.chunk_from_row(result)

    def get_chunks_by_path(self, path: Union[str, Path]) -> List[Dict[str, object]]:
        """Retrieve chunks by document path.

        Args:
            path: Path to the document

        Returns:
            List of chunk data dictionaries

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        results = self.conn.execute("""
            SELECT id, path, title, content, chunk_index, language, created_at, modified_at, metadata
            FROM chunks
            WHERE path = ?
            ORDER BY chunk_index
        """, [str(path)]).fetchall()

        chunk_data = []
        for result in results:
            chunk_data.append({
                "id": result[0],
                "path": result[1],
                "title": result[2],
                "content": result[3],
                "chunk_index": result[4],
                "language": result[5],
                "created_at": result[6],
                "modified_at": result[7],
                "metadata": json.loads(result[8]) if result[8] else {},
            })

        return chunk_data

    def delete_chunks_by_path(self, path: Union[str, Path]) -> int:
        """Delete chunks and their embeddings by document path.

        Args:
            path: Path to the document

        Returns:
            Number of chunks deleted

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # First get the IDs of chunks to delete
        chunk_ids = self.conn.execute("""
            SELECT id FROM chunks WHERE path = ?
        """, [str(path)]).fetchall()

        if not chunk_ids:
            return 0

        # Flatten the list of tuples
        chunk_id_list = [id[0] for id in chunk_ids]

        # Delete related embeddings first (to maintain foreign key integrity)
        for chunk_id in chunk_id_list:
            self.conn.execute("DELETE FROM embeddings WHERE chunk_id = ?", [chunk_id])

        # Then delete the chunks
        result = self.conn.execute("""
            DELETE FROM chunks WHERE path = ?
        """, [str(path)]).fetchone()

        # Handle the case where result might be None
        if result is None:
            return 0

        # Cast the first element of the result to int
        deleted_count: int = int(result[0])
        return deleted_count

    def backup(self, backup_path: Union[str, Path]) -> None:
        """Create a backup of the database.

        Args:
            backup_path: Path to store the backup

        """
        # Close connection to ensure all data is written
        if self.conn:
            self.conn.close()
            self.conn = None

        # Create backup
        shutil.copy2(self.db_path, backup_path)

        # Reconnect
        self.setup()

    def clear(self) -> None:
        """Clear all data from the database.

        This method removes all chunks and embeddings from the database
        while preserving the database schema and structure.
        """
        # Use QueryBuilder for safe data clearing
        clear_queries = QueryBuilder.clear_all_data()
        
        with self.db_manager.transaction() as conn:
            for sql, params in clear_queries:
                conn.execute(sql, params)

        # Recreate HNSW index to ensure clean state
        self.db_manager.index_manager.recreate_hnsw_index()

    def get_statistics(self) -> Dict[str, object]:
        """Retrieve statistics about the database.

        Returns:
            Dictionary with database statistics including document count,
            chunk count, languages, embedding model, and last updated timestamp

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # Get chunk count
        chunk_count_result = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        chunk_count = chunk_count_result[0] if chunk_count_result is not None else 0

        # Get document count (unique paths)
        document_count_result = self.conn.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()
        document_count = document_count_result[0] if document_count_result is not None else 0

        # Get available languages
        languages_result = self.conn.execute(
            "SELECT DISTINCT language FROM chunks WHERE language IS NOT NULL"
        ).fetchall()
        languages = [lang[0] for lang in languages_result if lang[0]]

        # Get embedding model
        model_result = self.conn.execute(
            "SELECT model FROM embeddings LIMIT 1"
        ).fetchone()
        model = model_result[0] if model_result else "unknown"

        # Get last updated timestamp
        last_updated_result = self.conn.execute(
            "SELECT MAX(modified_at) FROM chunks"
        ).fetchone()
        last_updated = last_updated_result[0] if last_updated_result and last_updated_result[0] else "unknown"

        # Return statistics
        return {
            "document_count": document_count,
            "chunk_count": chunk_count,
            "languages": languages,
            "embedding_model": model,
            "db_path": str(self.db_path),
            "last_updated": last_updated
        }

    def store_bm25_index(
        self,
        vocabulary: Dict[str, Tuple[int, int]],
        inverted_index: Dict[str, List[Tuple[str, int, List[int]]]],
        document_stats: Dict[str, Tuple[int, int, float]],
        collection_stats: Dict[str, Union[int, float]],
        batch_size: int = 10000,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Store BM25 index data in the database with batch processing.
        
        Args:
            vocabulary: Dictionary mapping terms to (doc_freq, collection_freq)
            inverted_index: Dictionary mapping terms to list of (chunk_id, term_freq, positions)
            document_stats: Dictionary mapping chunk_id to (total_terms, unique_terms, avg_term_freq)
            collection_stats: Dictionary with collection-level statistics
            batch_size: Number of records to process in each batch (default: 10000)
            progress_callback: Optional callback function for progress reporting

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")
        
        logger.info(f"Storing BM25 index with batch size: {batch_size}")
        logger.info(f"Vocabulary size: {len(vocabulary)}, Inverted index entries: {sum(len(postings) for postings in inverted_index.values())}")
        
        # Optimize DuckDB settings for bulk inserts
        logger.info("Optimizing database settings for bulk operations...")
        try:
            self.conn.execute("SET memory_limit='6GB'")
            self.conn.execute("SET threads=8")  # Use more threads for parallel processing
            self.conn.execute("SET max_memory='6GB'")
            logger.info("Applied memory and thread optimizations")
        except Exception as e:
            logger.warning(f"Could not apply some database optimizations: {e}")
            # Try more conservative settings
            try:
                self.conn.execute("SET threads=4")
                logger.info("Applied conservative thread settings")
            except Exception:
                logger.warning("Could not apply thread optimizations")
        
        # Start a transaction for bulk inserts
        self.conn.begin()
        
        try:
            # Store vocabulary with batch processing
            
            vocab_data = [
                (term, doc_freq, coll_freq)
                for term, (doc_freq, coll_freq) in vocabulary.items()
            ]
            
            if vocab_data:  # Only execute if there's data
                logger.info(f"Storing {len(vocab_data)} vocabulary terms in batches...")
                if progress_callback:
                    progress_callback("vocabulary", 0, len(vocab_data))
                
                # Check if vocabulary table is empty for bulk insert optimization
                vocab_count_result = self.conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()
                vocab_is_initial = vocab_count_result is not None and vocab_count_result[0] == 0
                
                if vocab_is_initial and len(vocab_data) > 1000:
                    # Use bulk VALUES insert for initial vocabulary (much faster)
                    vocab_batch_size = 25000  # Smaller batch for vocabulary
                    for i in range(0, len(vocab_data), vocab_batch_size):
                        batch = vocab_data[i:i + vocab_batch_size]
                        if i > 0 and i % vocab_batch_size == 0:
                            logger.debug(f"Vocabulary progress: {i}/{len(vocab_data)} terms")
                        
                        # Build bulk VALUES clause
                        values_list = []
                        for term, doc_freq, coll_freq in batch:
                            term_escaped = term.replace("'", "''") if term else ''
                            values_list.append(f"('{term_escaped}', {doc_freq}, {coll_freq})")
                        
                        values_clause = ",".join(values_list)
                        bulk_query = f"""
                            INSERT INTO vocabulary (term, document_frequency, collection_frequency)
                            VALUES {values_clause}
                        """
                        self.conn.execute(bulk_query)
                        
                        # Report progress after each batch
                        if progress_callback:
                            progress_callback("vocabulary", min(i + vocab_batch_size, len(vocab_data)), len(vocab_data))
                else:
                    # Use executemany for updates or small datasets
                    for i in range(0, len(vocab_data), batch_size):
                        batch = vocab_data[i:i + batch_size]
                        if i > 0 and i % (batch_size * 10) == 0:
                            logger.debug(f"Vocabulary progress: {i}/{len(vocab_data)} terms")
                        self.conn.executemany("""
                            INSERT INTO vocabulary (term, document_frequency, collection_frequency)
                            VALUES (?, ?, ?)
                            ON CONFLICT (term) DO UPDATE SET
                                document_frequency = excluded.document_frequency,
                                collection_frequency = excluded.collection_frequency
                        """, batch)
                        
                        # Report progress after each batch
                        if progress_callback:
                            progress_callback("vocabulary", min(i + batch_size, len(vocab_data)), len(vocab_data))
            
            
            # Store inverted index with optimized batch size
            
            # Prepare data for bulk insert
            inv_index_data = []
            for term, postings in inverted_index.items():
                for chunk_id, term_freq, positions in postings:
                    # Store positions as integer array (None when positions not stored)
                    inv_index_data.append((term, chunk_id, term_freq, positions))
            
            # Skip sorting for bulk insert - sorting 1M+ entries is expensive
            # inv_index_data.sort(key=lambda x: (x[0], x[1]))
            
            # Check if this is initial indexing (empty table)
            count_result = self.conn.execute("SELECT COUNT(*) FROM inverted_index").fetchone()
            is_initial = count_result is not None and count_result[0] == 0
            
            if inv_index_data:  # Only process if there's data
                logger.info(f"Storing {len(inv_index_data)} inverted index entries...")
                if progress_callback:
                    progress_callback("inverted_index", 0, len(inv_index_data))
                    
                if is_initial:
                    # For initial indexing, use bulk VALUES INSERT for maximum speed
                    # For initial indexing, use bulk VALUES INSERT for maximum speed
                    initial_batch_size = 50000  # Smaller batch for bulk VALUES to avoid query limits
                    
                    for i in range(0, len(inv_index_data), initial_batch_size):
                        inv_batch = inv_index_data[i:i + initial_batch_size]
                        if i > 0 and i % initial_batch_size == 0:
                            logger.debug(f"Inverted index progress: {i}/{len(inv_index_data)} entries")
                        
                        # Build bulk VALUES clause for much faster insertion
                        values_list = []
                        for term, chunk_id, term_freq, positions in inv_batch:
                            # Escape single quotes in strings
                            term_escaped = term.replace("'", "''") if term else ''
                            chunk_id_escaped = chunk_id.replace("'", "''") if chunk_id else ''
                            
                            # Handle positions array - can be None or empty list
                            if positions is None or len(positions) == 0:
                                pos_str = "NULL"
                            else:
                                pos_str = "[" + ",".join(map(str, positions)) + "]"
                            
                            values_list.append(f"('{term_escaped}', '{chunk_id_escaped}', {term_freq}, {pos_str})")
                        
                        # Execute bulk insert with VALUES clause
                        values_clause = ",".join(values_list)
                        bulk_query = f"""
                            INSERT INTO inverted_index (term, chunk_id, term_frequency, positions)
                            VALUES {values_clause}
                        """
                        self.conn.execute(bulk_query)
                        
                        
                        # Report progress after each batch
                        if progress_callback:
                            progress_callback("inverted_index", min(i + initial_batch_size, len(inv_index_data)), len(inv_index_data))
                    
                
                    # Create index after bulk insert for better performance
                    logger.info("Creating database indexes for search optimization...")
                    if progress_callback:
                        progress_callback("creating_indexes", 0, 2)
                    
                    self.conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_inverted_index_term
                        ON inverted_index(term)
                    """)
                    if progress_callback:
                        progress_callback("creating_indexes", 1, 2)
                    
                    self.conn.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_inverted_index_term_chunk
                        ON inverted_index(term, chunk_id)
                    """)
                    if progress_callback:
                        progress_callback("creating_indexes", 2, 2)
                else:
                    # For updates, consider using faster approaches
                    # For updates, use optimized strategy for large datasets
                    
                    # Strategy: Delete existing entries first, then bulk insert
                    # This is much faster than ON CONFLICT for large updates
                    if len(inv_index_data) > 100000:  # For large updates
                        
                        # Get unique chunk_ids being updated
                        chunk_ids = set(chunk_id for _, chunk_id, _, _ in inv_index_data)
                        chunk_id_list = "', '".join(cid.replace("'", "''") for cid in chunk_ids)
                        
                        # Delete existing entries for these chunks
                        self.conn.execute(f"DELETE FROM inverted_index WHERE chunk_id IN ('{chunk_id_list}')")
                        
                        # Now use bulk INSERT (no conflicts possible)
                        bulk_batch_size = 50000
                        for i in range(0, len(inv_index_data), bulk_batch_size):
                            inv_batch = inv_index_data[i:i + bulk_batch_size]
                            
                            # Build bulk VALUES clause
                            values_list = []
                            for term, chunk_id, term_freq, positions in inv_batch:
                                term_escaped = term.replace("'", "''") if term else ''
                                chunk_id_escaped = chunk_id.replace("'", "''") if chunk_id else ''
                                
                                if positions is None or len(positions) == 0:
                                    pos_str = "NULL"
                                else:
                                    pos_str = "[" + ",".join(map(str, positions)) + "]"
                                
                                values_list.append(f"('{term_escaped}', '{chunk_id_escaped}', {term_freq}, {pos_str})")
                            
                            # Execute bulk insert
                            values_clause = ",".join(values_list)
                            bulk_query = f"""
                                INSERT INTO inverted_index (term, chunk_id, term_frequency, positions)
                                VALUES {values_clause}
                            """
                            self.conn.execute(bulk_query)
                            
                            
                            # Report progress
                            if progress_callback:
                                progress_callback("inverted_index", min(i + bulk_batch_size, len(inv_index_data)), len(inv_index_data))
                        
                        
                    else:
                        # For smaller updates, use traditional ON CONFLICT
                        update_batch_size = batch_size  # Use configured batch size for updates
                        
                        for i in range(0, len(inv_index_data), update_batch_size):
                            inv_batch = inv_index_data[i:i + update_batch_size]
                            if i > 0 and i % (update_batch_size * 10) == 0:
                                logger.debug(f"Inverted index update progress: {i}/{len(inv_index_data)} entries")
                            
                            self.conn.executemany("""
                                INSERT INTO inverted_index (term, chunk_id, term_frequency, positions)
                                VALUES (?, ?, ?, ?)
                                ON CONFLICT (term, chunk_id) DO UPDATE SET
                                    term_frequency = excluded.term_frequency,
                                    positions = excluded.positions
                            """, inv_batch)
                            
                            
                            # Report progress after each batch
                            if progress_callback:
                                progress_callback("inverted_index", min(i + update_batch_size, len(inv_index_data)), len(inv_index_data))
                        
            
            
            # Store document statistics with batch processing
            
            doc_stats_data = [
                (chunk_id, total_terms, unique_terms, avg_freq)
                for chunk_id, (total_terms, unique_terms, avg_freq) in document_stats.items()
            ]
            
            if doc_stats_data:  # Only execute if there's data
                logger.info(f"Storing {len(doc_stats_data)} document statistics...")
                if progress_callback:
                    progress_callback("document_stats", 0, len(doc_stats_data))
                
                # Check if document_stats table is empty for bulk insert optimization
                doc_stats_count_result = self.conn.execute("SELECT COUNT(*) FROM document_stats").fetchone()
                doc_stats_is_initial = doc_stats_count_result is not None and doc_stats_count_result[0] == 0
                
                if doc_stats_is_initial and len(doc_stats_data) > 1000:
                    # Use bulk VALUES insert for initial document stats (much faster)
                    doc_stats_batch_size = 25000  # Large batch for document stats
                    for i in range(0, len(doc_stats_data), doc_stats_batch_size):
                        doc_batch = doc_stats_data[i:i + doc_stats_batch_size]
                        if i > 0 and i % doc_stats_batch_size == 0:
                            logger.debug(f"Document stats progress: {i}/{len(doc_stats_data)} documents")
                        
                        # Build bulk VALUES clause
                        values_list = []
                        for chunk_id, total_terms, unique_terms, avg_freq in doc_batch:
                            chunk_id_escaped = chunk_id.replace("'", "''") if chunk_id else ''
                            values_list.append(f"('{chunk_id_escaped}', {total_terms}, {unique_terms}, {avg_freq})")
                        
                        values_clause = ",".join(values_list)
                        bulk_query = f"""
                            INSERT INTO document_stats (chunk_id, total_terms, unique_terms, avg_term_frequency)
                            VALUES {values_clause}
                        """
                        self.conn.execute(bulk_query)
                        
                        # Report progress after each batch
                        if progress_callback:
                            progress_callback("document_stats", min(i + doc_stats_batch_size, len(doc_stats_data)), len(doc_stats_data))
                else:
                    # Use executemany for updates or small datasets
                    for i in range(0, len(doc_stats_data), batch_size):
                        doc_batch = doc_stats_data[i:i + batch_size]
                        if i > 0 and i % (batch_size * 10) == 0:
                            logger.debug(f"Document stats progress: {i}/{len(doc_stats_data)} documents")
                        self.conn.executemany("""
                            INSERT INTO document_stats (chunk_id, total_terms, unique_terms, avg_term_frequency)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                total_terms = excluded.total_terms,
                                unique_terms = excluded.unique_terms,
                                avg_term_frequency = excluded.avg_term_frequency
                        """, doc_batch)
                        
                        # Report progress after each batch
                        if progress_callback:
                            progress_callback("document_stats", min(i + batch_size, len(doc_stats_data)), len(doc_stats_data))
            
            
            # Store collection statistics
            self.conn.execute("""
            INSERT INTO collection_stats
            (id, total_documents, total_terms, avg_document_length, last_updated)
            VALUES (1, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (id) DO UPDATE SET
                total_documents = excluded.total_documents,
                total_terms = excluded.total_terms,
                avg_document_length = excluded.avg_document_length,
                last_updated = excluded.last_updated
            """, (
                collection_stats.get("total_documents", 0),
                collection_stats.get("total_terms", 0),
                collection_stats.get("avg_document_length", 0.0),
            ))
            self.conn.commit()
            
        except Exception as e:
            self.conn.rollback()
            raise e
        finally:
            # Ensure any remaining data is written
            pass
    
    def search_bm25(
        self,
        query_terms: List[str],
        limit: int = 10,
        language: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Search using BM25 scoring.
        
        Args:
            query_terms: List of query terms
            limit: Maximum number of results to return
            language: Optional language filter
            
        Returns:
            List of matching document chunks with BM25 scores

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")
        
        if not query_terms:
            return []
        
        # Build the query dynamically based on number of terms
        # This query calculates BM25 scores using the stored statistics
        terms_placeholder = ",".join(["?"] * len(query_terms))
        
        # Optimized query that avoids GROUP BY on large text fields
        base_query = f"""
        WITH query_terms AS (
            SELECT value AS term FROM (VALUES {",".join(f"('{term}')" for term in query_terms)})
        ),
        collection_info AS (
            SELECT
                total_documents,
                avg_document_length
            FROM collection_stats
            WHERE id = 1
        ),
        -- Calculate scores efficiently using only chunk_id
        chunk_scores AS (
            SELECT
                ii.chunk_id,
                SUM(
                    -- IDF component
                    LOG((ci.total_documents - v.document_frequency + 0.5) /
                        (v.document_frequency + 0.5) + 1.0) *
                    -- Normalized TF component (k1=1.2, b=0.75)
                    ((ii.term_frequency * 2.2) /
                     (ii.term_frequency + 1.2 *
                      (0.25 + 0.75 * (ds.total_terms / ci.avg_document_length))))
                ) AS score
            FROM inverted_index ii
            JOIN vocabulary v ON ii.term = v.term
            JOIN document_stats ds ON ii.chunk_id = ds.chunk_id
            CROSS JOIN collection_info ci
            WHERE ii.term IN ({terms_placeholder})
            GROUP BY ii.chunk_id
            ORDER BY score DESC
            LIMIT ?
        )
        -- Join with chunks table only for the top-k results
        SELECT
            c.id as chunk_id,
            c.path,
            c.title,
            c.content,
            c.chunk_index,
            c.language,
            c.metadata,
            cs.score
        FROM chunk_scores cs
        JOIN chunks c ON cs.chunk_id = c.id
        """
        
        # Add language filter if specified
        if language:
            base_query += f" WHERE c.language = '{language}'"
        
        base_query += " ORDER BY cs.score DESC"
        
        # Execute query
        params = query_terms + [limit]
        results = self.conn.execute(base_query, params).fetchall()
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                "chunk_id": row[0],
                "path": row[1],
                "title": row[2],
                "content": row[3],
                "chunk_index": row[4],
                "language": row[5],
                "metadata": json.loads(row[6]) if row[6] else {},
                "score": float(row[7]),  # BM25 score
            })
        
        return formatted_results
    
    def clear_bm25_index(self) -> None:
        """Clear all BM25 index data."""
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")
        
        # Clear BM25 tables
        self.conn.execute("DELETE FROM inverted_index")
        self.conn.execute("DELETE FROM vocabulary")
        self.conn.execute("DELETE FROM document_stats")
        self.conn.execute("DELETE FROM collection_stats")
        self.conn.commit()

    def _hnsw_index_exists(self) -> bool:
        """Check if HNSW index exists."""
        if self.conn is None:
            return False
        
        try:
            # Query DuckDB's internal catalog for indexes
            result = self.conn.execute("""
                SELECT COUNT(*)
                FROM duckdb_indexes
                WHERE index_name = 'vector_idx'
            """).fetchone()
            return bool(result and result[0] > 0)
        except Exception:
            # If the query fails, assume index doesn't exist
            return False
    
    def _create_hnsw_index(self) -> None:
        """Create HNSW index with configured parameters."""
        if self.conn is None:
            raise ValueError("Database connection not initialized.")
        
        self.conn.execute(f"""
            CREATE INDEX vector_idx ON embeddings
            USING HNSW (vector)
            WITH (
                metric = 'cosine',
                ef_construction = {self.ef_construction},
                ef_search = {self.ef_search},
                m = {self.m},
                m0 = {self.m0}
            )
        """)
    
    def _validate_hnsw_index_params(self) -> bool:
        """Validate that existing HNSW index has expected parameters.
        
        Returns:
            True if parameters match, False otherwise
        
        Note: DuckDB may not expose all index parameters for validation.
        This is a placeholder for when such functionality becomes available.
        For now, assume index is valid if it exists.

        """
        # TODO: Implement parameter validation when DuckDB exposes index metadata
        # For now, always return True to use existing index
        return True
    
    def recreate_hnsw_index(self, force: bool = False) -> None:
        """Recreate the HNSW index.
        
        Args:
            force: If True, recreate even if index exists
        
        This should be called when:
        - Index parameters need to be changed
        - Index corruption is suspected
        - After bulk data modifications

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")
        
        if force or self._hnsw_index_exists():
            logger.info("Dropping existing HNSW index...")
            self.conn.execute("DROP INDEX IF EXISTS vector_idx")
        
        logger.info("Creating new HNSW index...")
        self._create_hnsw_index()
        
        # Optionally compact after recreation
        self.recompact_index()
        logger.info("HNSW index recreation complete")
    
    def close(self) -> None:
        """Close the database connections."""
        # Use DatabaseManager for proper cleanup
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close()
        
        # Also handle direct connection for backward compatibility
        if hasattr(self, '_conn') and self._conn:
            self._conn.close()
            self._conn = None

