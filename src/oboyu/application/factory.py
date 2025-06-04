"""Factory for creating hexagonal architecture components."""

import logging

from ..adapters.config.configuration_adapter import ConfigurationAdapter
from ..adapters.filesystem.local_filesystem_adapter import LocalFilesystemAdapter
from ..application.container import Container
from ..interfaces.cli.hexagonal_facade import HexagonalFacade

logger = logging.getLogger(__name__)


class HexagonalArchitectureFactory:
    """Factory for creating hexagonal architecture components."""
    
    @staticmethod
    def create_container(database_service, embedding_service,
                        crawler_service, language_detector,
                        config_service, reranker_service=None) -> Container:
        """Create a fully configured dependency injection container."""
        container = Container()
        
        config_adapter = ConfigurationAdapter(config_service)
        container.register_configuration(config_adapter)
        
        filesystem_adapter = LocalFilesystemAdapter(crawler_service, language_detector)
        
        container.configure_default_services(
            database_service=database_service,
            embedding_service=embedding_service,
            filesystem_port=filesystem_adapter,
            reranker_service=reranker_service
        )
        
        return container
    
    @staticmethod
    def create_facade(database_service, embedding_service,
                     crawler_service, language_detector,
                     config_service, reranker_service=None) -> HexagonalFacade:
        """Create a hexagonal facade for backward compatibility."""
        container = HexagonalArchitectureFactory.create_container(
            database_service=database_service,
            embedding_service=embedding_service,
            crawler_service=crawler_service,
            language_detector=language_detector,
            config_service=config_service,
            reranker_service=reranker_service
        )
        
        return HexagonalFacade(container)
    
    @staticmethod
    def create_from_existing_services(indexer, retriever) -> HexagonalFacade:
        """Create hexagonal facade from existing indexer and retriever services."""
        return HexagonalArchitectureFactory.create_facade(
            database_service=indexer.database_service,
            embedding_service=indexer.embedding_service,
            crawler_service=indexer.crawler,
            language_detector=indexer.language_detector,
            config_service=indexer.config,
            reranker_service=getattr(retriever, 'reranker_service', None)
        )
