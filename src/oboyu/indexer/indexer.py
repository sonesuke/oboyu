"""New modular indexer facade implementation."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from oboyu.crawler.crawler import CrawlerResult
from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.core.document_processor import DocumentProcessor
from oboyu.indexer.core.search_engine import SearchEngine, SearchMode
from oboyu.indexer.models.embedding_service import EmbeddingService
from oboyu.indexer.models.reranker_service import RerankerService
from oboyu.indexer.models.tokenizer_service import TokenizerService
from oboyu.indexer.search.bm25_indexer import BM25Indexer
from oboyu.indexer.search.bm25_search import BM25Search
from oboyu.indexer.search.hybrid_search import HybridSearch
from oboyu.indexer.search.search_result import SearchResult
from oboyu.indexer.search.vector_search import VectorSearch
from oboyu.indexer.storage.change_detector import FileChangeDetector
from oboyu.indexer.storage.database_service import DatabaseService

logger = logging.getLogger(__name__)


class Indexer:
    """Lightweight facade class that coordinates modular services."""

    def __init__(self, config: Optional[IndexerConfig] = None) -> None:
        """Initialize the indexer with modular services.

        Args:
            config: Indexer configuration

        """
        # Initialize configuration
        self.config = config or IndexerConfig()

        # Initialize core services
        self._initialize_services()

        # Initialize search engine
        self._initialize_search_engine()

        # Initialize change detector for persistent tracking
        self.change_detector = FileChangeDetector(self.database_service)

    def _initialize_services(self) -> None:
        """Initialize all modular services."""
        # Ensure config is properly initialized
        assert self.config.processing is not None, "ProcessingConfig should be initialized"
        assert self.config.model is not None, "ModelConfig should be initialized"
        assert self.config.search is not None, "SearchConfig should be initialized"
        
        # Document processor
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.processing.chunk_size,
            chunk_overlap=self.config.processing.chunk_overlap,
            document_prefix=self.config.model.document_prefix,
        )

        # Embedding service
        self.embedding_service = EmbeddingService(
            model_name=self.config.model.embedding_model,
            device=self.config.model.embedding_device,
            batch_size=self.config.model.batch_size,
            max_seq_length=self.config.model.max_seq_length,
            query_prefix=self.config.model.query_prefix,
            use_onnx=self.config.model.use_onnx,
            onnx_quantization_config=self.config.model.onnx_quantization,
            onnx_optimization_level=self.config.model.onnx_optimization_level,
        )

        # Database service
        embedding_dims = self.embedding_service.dimensions or 256

        # Create HNSW parameters
        from oboyu.indexer.storage.index_manager import HNSWIndexParams

        hnsw_params = HNSWIndexParams(
            ef_construction=self.config.processing.ef_construction,
            ef_search=self.config.processing.ef_search,
            m=self.config.processing.m,
            m0=self.config.processing.m0,
        )

        self.database_service = DatabaseService(
            db_path=self.config.processing.db_path,
            embedding_dimensions=embedding_dims,
            hnsw_params=hnsw_params,
        )

        # Initialize database
        self.database_service.initialize()

        # BM25 indexer
        self.bm25_indexer = BM25Indexer(
            k1=self.config.search.bm25_k1,
            b=self.config.search.bm25_b,
            tokenizer_kwargs={
                "min_token_length": self.config.search.bm25_min_token_length,
            },
            use_stopwords=True,
            min_doc_frequency=2,
            store_positions=False,
        )

        # Tokenizer service
        self.tokenizer_service = TokenizerService(
            language="ja",  # Default to Japanese
            tokenizer_kwargs={
                "min_token_length": self.config.search.bm25_min_token_length,
            },
        )

        # Reranker service (optional)
        self.reranker_service = None
        if self.config.use_reranker:
            self.reranker_service = RerankerService(
                model_name=self.config.model.reranker_model,
                use_onnx=self.config.model.reranker_use_onnx,
                device=self.config.model.reranker_device,
                batch_size=self.config.model.reranker_batch_size,
                max_length=self.config.model.reranker_max_length,
                quantization_config=self.config.model.onnx_quantization,
                optimization_level=self.config.model.onnx_optimization_level,
            )

    def _initialize_search_engine(self) -> None:
        """Initialize the search engine with search services."""
        # Ensure config is properly initialized
        assert self.config.search is not None, "SearchConfig should be initialized"
        
        # Vector search
        vector_search = VectorSearch(self.database_service)

        # BM25 search
        bm25_search = BM25Search(self.database_service)

        # Hybrid search
        hybrid_search = HybridSearch(
            vector_weight=self.config.search.vector_weight,
            bm25_weight=self.config.search.bm25_weight,
        )

        # Search engine coordinator
        self.search_engine = SearchEngine(
            vector_search=vector_search,
            bm25_search=bm25_search,
            hybrid_search=hybrid_search,
        )

    def index_documents(self, crawler_results: List[CrawlerResult], progress_callback: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
        """Index documents using the coordinated services.

        Args:
            crawler_results: Results from document crawling
            progress_callback: Optional callback for progress updates

        Returns:
            Indexing result summary

        """
        if not crawler_results:
            return {"indexed_chunks": 0, "total_documents": 0}

        try:
            # Process documents into chunks with progress tracking
            all_chunks = []
            total_docs = len(crawler_results)
            
            if progress_callback:
                progress_callback("processing", 0, total_docs)
                
            for i, result in enumerate(crawler_results):
                chunks = self.document_processor.process_document(
                    path=result.path,
                    content=result.content,
                    title=result.title,
                    language=result.language,
                    metadata=result.metadata,
                )
                all_chunks.extend(chunks)
                
                if progress_callback:
                    progress_callback("processing", i + 1, total_docs)

            if not all_chunks:
                return {"indexed_chunks": 0, "total_documents": len(crawler_results)}

            # Store chunks in database with progress tracking
            total_chunks = len(all_chunks)
            if progress_callback:
                progress_callback("storing", 0, total_chunks)
            
            self.database_service.store_chunks(all_chunks, progress_callback=progress_callback)

            # Generate embeddings with progress tracking
            texts_for_embedding = self.document_processor.prepare_for_embedding(all_chunks)
            logger.info(f"Generating embeddings for {len(texts_for_embedding)} text chunks...")
            
            if progress_callback:
                progress_callback("embedding", 0, len(texts_for_embedding))
                
            embeddings = self.embedding_service.generate_embeddings(texts_for_embedding, progress_callback=progress_callback)

            # Store embeddings with progress tracking
            chunk_ids = [chunk.id for chunk in all_chunks]
            if progress_callback:
                progress_callback("storing_embeddings", 0, len(chunk_ids))
                
            self.database_service.store_embeddings(chunk_ids, embeddings, progress_callback=progress_callback)
            
            # Ensure HNSW index exists after storing embeddings
            self.database_service.ensure_hnsw_index()

            # Update BM25 index with progress tracking
            if progress_callback:
                progress_callback("bm25_indexing", 0, 5)  # BM25 has 5 main steps
                
            self.bm25_indexer.index_chunks(all_chunks, progress_callback=progress_callback)

            return {
                "indexed_chunks": len(all_chunks),
                "total_documents": len(crawler_results),
            }

        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return {"indexed_chunks": 0, "total_documents": len(crawler_results), "error": str(e)}

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
        language_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search using the coordinated search engine.

        Args:
            query: Search query
            limit: Maximum number of results
            mode: Search mode ("vector", "bm25", "hybrid")
            language_filter: Optional language filter

        Returns:
            List of search results

        """
        try:
            # Ensure config is properly initialized
            assert self.config.search is not None, "SearchConfig should be initialized"
            
            # Convert mode string to enum
            if mode == "vector":
                search_mode = SearchMode.VECTOR
            elif mode == "bm25":
                search_mode = SearchMode.BM25
            elif mode == "hybrid":
                search_mode = SearchMode.HYBRID
            else:
                logger.warning(f"Unknown search mode: {mode}, using hybrid")
                search_mode = SearchMode.HYBRID

            # Prepare search inputs
            query_vector = None
            query_terms = None

            if search_mode in [SearchMode.VECTOR, SearchMode.HYBRID]:
                query_vector = self.embedding_service.generate_query_embedding(query)

            if search_mode in [SearchMode.BM25, SearchMode.HYBRID]:
                query_terms = self.tokenizer_service.tokenize_query(query)

            # Execute search
            results = self.search_engine.search(
                query_vector=query_vector,
                query_terms=query_terms,
                mode=search_mode,
                limit=limit * self.config.search.top_k_multiplier if self.reranker_service else limit,
                language_filter=language_filter,
                top_k_multiplier=self.config.search.top_k_multiplier,
            )

            # Apply reranking if enabled
            if self.reranker_service and self.reranker_service.is_available() and results:
                results = self.reranker_service.rerank(query, results)

            # Return final results with limit
            return results[:limit]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def vector_search(
        self,
        query: Union[str, NDArray[np.float32]],
        top_k: int = 10,
        limit: Optional[int] = None,
        language_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Vector search using query string or embedding.

        Args:
            query: Search query string or pre-computed embedding
            top_k: Maximum number of results (new parameter name)
            limit: Deprecated parameter name for compatibility
            language_filter: Optional language filter

        Returns:
            List of search results

        """
        # Handle legacy parameter name
        result_limit = top_k if limit is None else limit
        
        # Handle both string queries and embedding vectors
        if isinstance(query, str):
            query_embedding = self.embedding_service.generate_query_embedding(query)
        else:
            query_embedding = query
            
        return self.search_engine.search(
            query_vector=query_embedding,
            mode=SearchMode.VECTOR,
            limit=result_limit,
            language_filter=language_filter,
        )

    def delete_document(self, path: Path) -> int:
        """Delete a document and all its chunks.

        Args:
            path: Path to the document to delete

        Returns:
            Number of chunks deleted

        """
        return self.database_service.delete_chunks_by_path(path)

    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics.

        Returns:
            Dictionary with indexer statistics

        """
        # Ensure config is properly initialized
        assert self.config.model is not None, "ModelConfig should be initialized"
        
        return {
            "total_chunks": self.database_service.get_chunk_count(),
            "indexed_paths": len(self.database_service.get_paths_with_chunks()),
            "embedding_model": self.config.model.embedding_model,
            "reranker_enabled": self.config.use_reranker,
        }

    def clear_index(self) -> None:
        """Clear all indexed data."""
        self.database_service.clear_database()
        if hasattr(self.bm25_indexer, "clear"):
            self.bm25_indexer.clear()

    def close(self) -> None:
        """Close all services and connections."""
        if hasattr(self.database_service, "close"):
            self.database_service.close()

    @classmethod
    def from_path(cls, db_path: Union[str, Path]) -> "Indexer":
        """Create an indexer from a database path.

        Args:
            db_path: Path to the database file

        Returns:
            Initialized Indexer instance

        """
        from oboyu.indexer.config.indexer_config import IndexerConfig
        
        config = IndexerConfig()
        config.db_path = db_path  # Use the property which handles the assertion
        return cls(config)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics

        """
        return self.database_service.get_database_stats()

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """BM25 search using query string.

        Args:
            query: Search query string
            top_k: Maximum number of results
            language_filter: Optional language filter

        Returns:
            List of search results

        """
        # Tokenize the query
        query_terms = self.tokenizer_service.tokenize_query(query)
        
        # Execute BM25 search
        return self.search_engine.search(
            query_terms=query_terms,
            mode=SearchMode.BM25,
            limit=top_k,
            language_filter=language_filter,
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        language_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Hybrid search using query string.

        Args:
            query: Search query string
            top_k: Maximum number of results
            vector_weight: Weight for vector search component
            bm25_weight: Weight for BM25 search component
            language_filter: Optional language filter

        Returns:
            List of search results

        """
        # Generate query embedding and tokenize query
        query_vector = self.embedding_service.generate_query_embedding(query)
        query_terms = self.tokenizer_service.tokenize_query(query)
        
        # Execute hybrid search
        return self.search_engine.search(
            query_vector=query_vector,
            query_terms=query_terms,
            mode=SearchMode.HYBRID,
            limit=top_k,
            language_filter=language_filter,
        )

    def rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results using the reranker service.

        Args:
            query: Original search query
            results: Search results to rerank

        Returns:
            Reranked search results

        """
        if not self.reranker_service or not self.reranker_service.is_available():
            logger.warning("Reranker service not available, returning original results")
            return results
            
        return self.reranker_service.rerank(query, results)
