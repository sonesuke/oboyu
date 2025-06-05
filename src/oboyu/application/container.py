"""Dependency injection container for the application."""

import logging
from typing import Any, Dict, Optional, Type, TypeVar

from ..adapters.database.duckdb_search_repository import DuckDBSearchRepository
from ..adapters.embedding.huggingface_embedding_service import HuggingFaceEmbeddingService
from ..application.indexing.indexing_service import IndexingService
from ..application.searching.search_service import SearchService
from ..domain.services.document_processor import DocumentProcessor
from ..domain.services.search_engine import SearchEngine
from ..ports.external.configuration_port import ConfigurationPort
from ..ports.external.filesystem_port import FilesystemPort
from ..ports.repositories.search_repository import SearchRepository
from ..ports.services.embedding_service import EmbeddingService
from ..ports.services.reranker_service import RerankerService

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Container:
    """Simple dependency injection container."""
    
    def __init__(self) -> None:
        """Initialize container."""
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._configuration: Optional[ConfigurationPort] = None
    
    def register_singleton(self, interface: Type[T], implementation: T) -> None:
        """Register a singleton service."""
        self._singletons[interface] = implementation
    
    def register_transient(self, interface: Type[T], factory: callable) -> None:
        """Register a transient service with factory."""
        self._services[interface] = factory
    
    def register_configuration(self, config: ConfigurationPort) -> None:
        """Register configuration port."""
        self._configuration = config
        self.register_singleton(ConfigurationPort, config)
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service from the container."""
        if interface in self._singletons:
            return self._singletons[interface]
        
        if interface in self._services:
            factory = self._services[interface]
            return factory()
        
        raise ValueError(f"Service {interface} not registered")
    
    def configure_default_services(self,
                                 database_service: Any,  # noqa: ANN401
                                 embedding_service: Any,  # noqa: ANN401
                                 filesystem_port: FilesystemPort,
                                 reranker_service: Optional[RerankerService] = None) -> None:
        """Configure default services for hexagonal architecture."""
        search_repository = DuckDBSearchRepository(database_service)
        self.register_singleton(SearchRepository, search_repository)
        
        hf_embedding_service = HuggingFaceEmbeddingService(embedding_service)
        self.register_singleton(EmbeddingService, hf_embedding_service)
        
        self.register_singleton(FilesystemPort, filesystem_port)
        
        if reranker_service:
            self.register_singleton(RerankerService, reranker_service)
        
        document_processor = DocumentProcessor()
        self.register_singleton(DocumentProcessor, document_processor)
        
        search_engine = SearchEngine()
        self.register_singleton(SearchEngine, search_engine)
        
        self.register_transient(
            IndexingService,
            lambda: IndexingService(
                search_repository=self.resolve(SearchRepository),
                embedding_service=self.resolve(EmbeddingService),
                filesystem_port=self.resolve(FilesystemPort),
                document_processor=self.resolve(DocumentProcessor)
            )
        )
        
        self.register_transient(
            SearchService,
            lambda: SearchService(
                search_repository=self.resolve(SearchRepository),
                embedding_service=self.resolve(EmbeddingService),
                search_engine=self.resolve(SearchEngine),
                reranker_service=self._singletons.get(RerankerService)
            )
        )
    
    def get_indexing_service(self) -> IndexingService:
        """Get indexing service."""
        return self.resolve(IndexingService)
    
    def get_search_service(self) -> SearchService:
        """Get search service."""
        return self.resolve(SearchService)
