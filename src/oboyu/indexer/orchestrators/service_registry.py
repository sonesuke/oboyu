"""Service registry for managing indexer dependencies."""

import logging

from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.indexer.core.document_processor import DocumentProcessor
from oboyu.indexer.services.embedding import EmbeddingService
from oboyu.indexer.storage.change_detector import FileChangeDetector
from oboyu.indexer.storage.database_service import DatabaseService
from oboyu.indexer.storage.index_manager import HNSWIndexParams
from oboyu.retriever.search.bm25_indexer import BM25Indexer
from oboyu.retriever.services.tokenizer import TokenizerService

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Manages service instances and their dependencies."""
    
    def __init__(self, config: IndexerConfig) -> None:
        """Initialize the service registry with configuration.
        
        Args:
            config: Indexer configuration
            
        """
        self.config = config
        self._services: dict = {}
        self._initialize_services()
        
    def _initialize_services(self) -> None:
        """Initialize all services with proper dependencies."""
        # Ensure config is properly initialized
        assert self.config.processing is not None, "ProcessingConfig should be initialized"
        assert self.config.model is not None, "ModelConfig should be initialized"
        assert self.config.search is not None, "SearchConfig should be initialized"
        
        # Initialize database service
        self._services["database_service"] = DatabaseService(
            db_path=self.config.processing.db_path,
            hnsw_params=HNSWIndexParams(),
        )
        
        # Initialize embedding service
        self._services["embedding_service"] = EmbeddingService(
            model_name=self.config.model.embedding_model,
            device=self.config.model.embedding_device,
            batch_size=self.config.model.embedding_batch_size,
            use_onnx=self.config.model.use_onnx,
            cache_dir=self.config.model.model_cache_dir,
            quantization_config=self.config.model.onnx_quantization,
            optimization_level=self.config.model.onnx_optimization_level,
        )
        
        # Initialize document processor
        self._services["document_processor"] = DocumentProcessor(
            embedding_service=self.get_embedding_service(),
            database_service=self.get_database_service(),
            chunk_size=self.config.processing.chunk_size,
            chunk_overlap=self.config.processing.chunk_overlap,
            max_file_size=self.config.processing.max_file_size,
        )
        
        # Initialize BM25 indexer
        self._services["bm25_indexer"] = BM25Indexer(
            language="ja",  # Default to Japanese
            bm25_params={
                "k1": self.config.search.bm25_k1,
                "b": self.config.search.bm25_b,
            },
            min_token_length=self.config.search.bm25_min_token_length,
            min_doc_frequency=2,
            store_positions=False,
        )
        
        # Initialize tokenizer service
        self._services["tokenizer_service"] = TokenizerService(
            language="ja",  # Default to Japanese
            tokenizer_kwargs={
                "min_token_length": self.config.search.bm25_min_token_length,
            },
        )
        
        # Initialize change detector
        self._services["change_detector"] = FileChangeDetector(self.get_database_service())
        
    def get_database_service(self) -> DatabaseService:
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
        
    def get_document_processor(self) -> DocumentProcessor:
        """Get the document processor instance.
        
        Returns:
            DocumentProcessor instance
            
        """
        return self._services["document_processor"]  # type: ignore
        
    def get_bm25_indexer(self) -> BM25Indexer:
        """Get the BM25 indexer instance.
        
        Returns:
            BM25Indexer instance
            
        """
        return self._services["bm25_indexer"]  # type: ignore
        
    def get_tokenizer_service(self) -> TokenizerService:
        """Get the tokenizer service instance.
        
        Returns:
            TokenizerService instance
            
        """
        return self._services["tokenizer_service"]  # type: ignore
        
    def get_change_detector(self) -> FileChangeDetector:
        """Get the change detector instance.
        
        Returns:
            FileChangeDetector instance
            
        """
        return self._services["change_detector"]  # type: ignore
