"""Factory for creating hexagonal architecture components."""

import logging
from typing import Any

from ..adapters.config.configuration_adapter import ConfigurationAdapter
from ..adapters.filesystem.local_filesystem_adapter import LocalFilesystemAdapter
from ..application.container import Container
from ..interfaces.cli.hexagonal_facade import HexagonalFacade

logger = logging.getLogger(__name__)


class HexagonalArchitectureFactory:
    """Factory for creating hexagonal architecture components."""
    
    @staticmethod
    def create_container(database_service: Any,  # noqa: ANN401
                        embedding_service: Any,  # noqa: ANN401
                        crawler_service: Any,  # noqa: ANN401
                        language_detector: Any,  # noqa: ANN401
                        config_service: Any,  # noqa: ANN401
                        reranker_service: Any = None) -> Container:  # noqa: ANN401
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
    def create_facade(database_service: Any,  # noqa: ANN401
                     embedding_service: Any,  # noqa: ANN401
                     crawler_service: Any,  # noqa: ANN401
                     language_detector: Any,  # noqa: ANN401
                     config_service: Any,  # noqa: ANN401
                     reranker_service: Any = None) -> HexagonalFacade:  # noqa: ANN401
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
    def create_from_existing_services(indexer: Any, retriever: Any) -> HexagonalFacade:  # noqa: ANN401
        """Create hexagonal facade from existing indexer and retriever services."""
        return HexagonalArchitectureFactory.create_facade(
            database_service=indexer.database_service,
            embedding_service=indexer.embedding_service,
            crawler_service=indexer.crawler,
            language_detector=indexer.language_detector,
            config_service=indexer.config,
            reranker_service=getattr(retriever, 'reranker_service', None)
        )
