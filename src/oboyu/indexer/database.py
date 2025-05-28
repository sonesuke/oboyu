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
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from oboyu.indexer.config import DEFAULT_BATCH_SIZE
from oboyu.indexer.processor import Chunk

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
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m = m
        self.m0 = m0 if m0 is not None else 2 * m
        self.batch_size = batch_size

        # Connections will be initialized in setup()
        self.conn: Optional[DuckDBPyConnection] = None


    def setup(self) -> None:
        """Set up the database schema and extensions."""
        # Ensure the parent directory exists
        if not str(self.db_path).startswith(":memory:"):
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Always use a file-based database to ensure persistence
        self.conn = duckdb.connect(str(self.db_path))

        # Install and load VSS extension first
        # The VSS extension is a system library that comes with DuckDB
        # See: https://duckdb.org/docs/stable/extensions/vss.html
        self.conn.execute("INSTALL vss")
        self.conn.execute("LOAD vss")

        # Enable experimental persistence for HNSW indexes
        self.conn.execute("SET hnsw_enable_experimental_persistence=true")
        
        # Set performance optimization pragmas
        self.conn.execute("PRAGMA threads=8")  # Increase thread count
        self.conn.execute("SET preserve_insertion_order=false")
        self.conn.execute("SET memory_limit='4GB'")  # Increase memory limit
        self.conn.execute("SET temp_directory='/tmp'")  # Use faster temp directory

        # Create tables if they don't exist
        self._create_schema()



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

        # Drop existing index if it exists
        self.conn.execute("DROP INDEX IF EXISTS vector_idx")

        # Create HNSW index with parameters
        # DuckDB doesn't support ? in DDL statements, use string formatting carefully
        # These are controlled inputs, not user input
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
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        if not chunks:
            return

        self._store_chunks_duckdb(chunks)


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
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        if not embeddings:
            return

        # Process embeddings in batches to control memory usage
        for batch_start in range(0, len(embeddings), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(embeddings))
            embeddings_batch = embeddings[batch_start:batch_end]
            self._store_embeddings_duckdb(embeddings_batch, model_name)

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
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # First create a temporary table with the query vector
        temp_table_name = f"temp_query_vector_{uuid.uuid4().hex}"

        # We need to ensure query_vector is a numpy float32 array
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        # Create a temporary table with the query vector
        self.conn.execute(f"CREATE TEMPORARY TABLE {temp_table_name}(vec FLOAT[{self.embedding_dimensions}])")

        # Insert the vector as a list
        vector_list = query_vector.tolist()
        self.conn.execute(f"INSERT INTO {temp_table_name} VALUES (?)", [vector_list])

        # Build the query using the temporary table
        sql_query = f"""
        SELECT
            c.id as chunk_id,
            c.path,
            c.title,
            c.content,
            c.chunk_index,
            c.language,
            c.metadata,
            array_distance(e.vector, q.vec) as score
        FROM chunks c
        JOIN embeddings e ON c.id = e.chunk_id
        CROSS JOIN {temp_table_name} q
        """

        # Add language filter if needed
        params = []
        if language:
            sql_query += " WHERE c.language = ? "
            params.append(language)

        # Add ordering and limit
        sql_query += f" ORDER BY score ASC LIMIT {limit}"

        # Execute search
        results = self.conn.execute(sql_query, params).fetchall()

        # Clean up temporary table
        self.conn.execute(f"DROP TABLE IF EXISTS {temp_table_name}")

        # Format results
        formatted_results = []
        for row in results:
            # Convert cosine distance to similarity score
            # Cosine distance ranges from 0 to 2, where 0 is best match
            # Convert to similarity score where 1 is best match
            distance = float(row[7])
            similarity_score = 1.0 - (distance / 2.0)
            
            formatted_results.append({
                "chunk_id": row[0],
                "path": row[1],
                "title": row[2],
                "content": row[3],
                "chunk_index": row[4],
                "language": row[5],
                "metadata": json.loads(row[6]) if row[6] else {},
                "score": similarity_score,  # Now higher scores are better
            })

        return formatted_results

    def recompact_index(self) -> None:
        """Recompact the HNSW index to improve search performance."""
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        self.conn.execute("PRAGMA hnsw_compact_index('vector_idx')")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, object]]:
        """Retrieve a chunk by its ID.

        Args:
            chunk_id: ID of the chunk to retrieve

        Returns:
            Chunk data as dictionary or None if not found

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        result = self.conn.execute("""
            SELECT id, path, title, content, chunk_index, language, created_at, modified_at, metadata
            FROM chunks
            WHERE id = ?
        """, [chunk_id]).fetchone()

        if not result:
            return None

        return {
            "id": result[0],
            "path": result[1],
            "title": result[2],
            "content": result[3],
            "chunk_index": result[4],
            "language": result[5],
            "created_at": result[6],
            "modified_at": result[7],
            "metadata": json.loads(result[8]) if result[8] else {},
        }

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
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")

        # Clear BM25 index data first (inverted_index references chunks)
        self.clear_bm25_index()
        
        # Delete all data from embeddings (due to foreign key constraint)
        self.conn.execute("DELETE FROM embeddings")

        # Delete all data from chunks
        self.conn.execute("DELETE FROM chunks")

        # Drop and recreate the HNSW index to ensure clean state
        # This is necessary because HNSW indexes can retain internal state
        # even after all data is deleted
        self.conn.execute("DROP INDEX IF EXISTS vector_idx")
        
        # Recreate the HNSW index with the same parameters
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
    ) -> None:
        """Store BM25 index data in the database with batch processing.
        
        Args:
            vocabulary: Dictionary mapping terms to (doc_freq, collection_freq)
            inverted_index: Dictionary mapping terms to list of (chunk_id, term_freq, positions)
            document_stats: Dictionary mapping chunk_id to (total_terms, unique_terms, avg_term_freq)
            collection_stats: Dictionary with collection-level statistics
            batch_size: Number of records to process in each batch (default: 10000)

        """
        if self.conn is None:
            raise ValueError("Database connection not initialized. Call setup() first.")
        
        logger.info(f"Storing BM25 index with batch size: {batch_size}")
        logger.info(f"Vocabulary size: {len(vocabulary)}, Inverted index entries: {sum(len(postings) for postings in inverted_index.values())}")
        
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
            
            
            # Store inverted index with optimized batch size
            
            # Prepare data for bulk insert
            inv_index_data = []
            for term, postings in inverted_index.items():
                for chunk_id, term_freq, positions in postings:
                    # Store positions as integer array (None for empty positions)
                    positions_array = positions if positions else None
                    inv_index_data.append((term, chunk_id, term_freq, positions_array))
            
            # Sort data by term for better insertion performance
            inv_index_data.sort(key=lambda x: (x[0], x[1]))
            
            # Check if this is initial indexing (empty table)
            count_result = self.conn.execute("SELECT COUNT(*) FROM inverted_index").fetchone()
            is_initial = count_result is not None and count_result[0] == 0
            
            if inv_index_data:  # Only process if there's data
                logger.info(f"Storing {len(inv_index_data)} inverted index entries...")
                if is_initial:
                    # For initial indexing, use simple INSERT for speed
                    # Use larger batches for better performance
                    initial_batch_size = 50000  # Larger batch size for initial indexing
                    for i in range(0, len(inv_index_data), initial_batch_size):
                        inv_batch = inv_index_data[i:i + initial_batch_size]
                        if i > 0 and i % (initial_batch_size * 5) == 0:
                            logger.debug(f"Inverted index progress: {i}/{len(inv_index_data)} entries")
                        self.conn.executemany("""
                            INSERT INTO inverted_index (term, chunk_id, term_frequency, positions)
                            VALUES (?, ?, ?, ?)
                        """, inv_batch)
                
                    # Create index after bulk insert for better performance
                    self.conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_inverted_index_term
                        ON inverted_index(term)
                    """)
                    self.conn.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_inverted_index_term_chunk
                        ON inverted_index(term, chunk_id)
                    """)
                else:
                    # For updates, use ON CONFLICT with batching
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
            
            
            # Store document statistics with batch processing
            
            doc_stats_data = [
                (chunk_id, total_terms, unique_terms, avg_freq)
                for chunk_id, (total_terms, unique_terms, avg_freq) in document_stats.items()
            ]
            
            if doc_stats_data:  # Only execute if there's data
                logger.info(f"Storing {len(doc_stats_data)} document statistics...")
                # Use same batch size as vocabulary
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

    def close(self) -> None:
        """Close the database connections."""
        # Close the DuckDB connection
        if self.conn:
            self.conn.close()
            self.conn = None

