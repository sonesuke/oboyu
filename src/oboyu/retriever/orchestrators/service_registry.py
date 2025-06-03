"""Service registry for managing retriever dependencies."""

import logging
from typing import TYPE_CHECKING, Optional

from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.services.embedding import EmbeddingService
from oboyu.indexer.storage.index_manager import HNSWIndexParams

if TYPE_CHECKING:
    from oboyu.indexer.storage.database_service import DatabaseService
from oboyu.common.services import TokenizerService
from oboyu.retriever.search.bm25_search import BM25Search
from oboyu.retriever.search.search_engine import SearchEngine
from oboyu.retriever.search.vector_search import VectorSearch
from oboyu.retriever.services.reranker import RerankerService

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Manages service instances and their dependencies for retrieval operations."""
    
    def __init__(self, config: IndexerConfig) -> None:
        """Initialize the service registry with configuration.
        
        Args:
            config: Indexer configuration (will be replaced with RetrieverConfig)
            
        """
        self.config = config
        self._services: dict[str, object] = {}
        self._initialize_services()
        
    def _initialize_services(self) -> None:
        """Initialize all services with proper dependencies."""
        # Ensure config is properly initialized
        assert self.config.processing is not None, "ProcessingConfig should be initialized"
        assert self.config.model is not None, "ModelConfig should be initialized"
        assert self.config.search is not None, "SearchConfig should be initialized"
        
        # Initialize embedding service (needed for query embeddings)
        self._services["embedding_service"] = EmbeddingService(
            model_name=self.config.model.embedding_model,
            batch_size=self.config.model.batch_size,
            max_seq_length=self.config.model.max_seq_length,
            query_prefix=self.config.model.query_prefix,
            use_onnx=self.config.model.use_onnx,
            onnx_quantization_config=self.config.model.onnx_quantization,
            onnx_optimization_level=self.config.model.onnx_optimization_level,
        )
        
        # Initialize database service (read-only for retrieval)
        embedding_dims = self.get_embedding_service().dimensions or 256
        hnsw_params = HNSWIndexParams(
            ef_construction=self.config.processing.ef_construction,
            ef_search=self.config.processing.ef_search,
            m=self.config.processing.m,
            m0=self.config.processing.m0,
        )
        
        # Import at runtime to avoid circular dependency
        from oboyu.indexer.storage.database_service import DatabaseService
        
        self._services["database_service"] = DatabaseService(
            db_path=self.config.processing.db_path,
            embedding_dimensions=embedding_dims,
            hnsw_params=hnsw_params,
        )
        
        # Initialize database
        self.get_database_service().initialize()
        
        # Initialize tokenizer service (for query tokenization)
        self._services["tokenizer_service"] = TokenizerService(
            language="ja",  # Default to Japanese
            tokenizer_kwargs={
                "min_token_length": self.config.search.bm25_min_token_length,
            },
        )
        
        # Initialize reranker service (optional)
        if self.config.use_reranker:
            self._services["reranker_service"] = RerankerService(
                model_name=self.config.model.reranker_model,
                use_onnx=self.config.model.reranker_use_onnx,
                batch_size=self.config.model.reranker_batch_size,
                max_length=self.config.model.reranker_max_length,
                quantization_config=self.config.model.onnx_quantization,
                optimization_level=self.config.model.onnx_optimization_level,
            )
        
        # Initialize search engine
        self._initialize_search_engine()
        
    def _initialize_search_engine(self) -> None:
        """Initialize the search engine with search services."""
        # Ensure config is properly initialized
        assert self.config.search is not None, "SearchConfig should be initialized"
        
        # Create search components
        vector_search = VectorSearch(self.get_database_service())
        bm25_search = BM25Search(self.get_database_service())
        
        # Create search engine
        self._services["search_engine"] = SearchEngine(
            vector_search=vector_search,
            bm25_search=bm25_search,
            rrf_k=self.config.search.rrf_k,
        )
        
    def get_database_service(self) -> "DatabaseService":
        """Get the database service instance.
        
        Returns:
            DatabaseService instance
            
        """
        return self._services["database_service"]  # type: ignore
        
    def get_embedding_service(self) -> EmbeddingService:
        """Get the embedding service instance.
        
        Returns:
            EmbeddingService instance
            
        """
        return self._services["embedding_service"]  # type: ignore
        
    def get_search_engine(self) -> SearchEngine:
        """Get the search engine instance.
        
        Returns:
            SearchEngine instance
            
        """
        return self._services["search_engine"]  # type: ignore
        
    def get_reranker_service(self) -> Optional[RerankerService]:
        """Get the reranker service instance if available.
        
        Returns:
            RerankerService instance or None
            
        """
        return self._services.get("reranker_service")  # type: ignore
        
    def get_tokenizer_service(self) -> TokenizerService:
        """Get the tokenizer service instance.
        
        Returns:
            TokenizerService instance
            
        """
        return self._services["tokenizer_service"]  # type: ignore
        
    def close(self) -> None:
        """Close all services that require cleanup."""
        database_service = self.get_database_service()
        if hasattr(database_service, "close"):
            database_service.close()
