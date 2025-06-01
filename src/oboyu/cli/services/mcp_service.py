"""MCP service for handling MCP server operations."""

from pathlib import Path
from typing import Literal, Optional

from oboyu.common.config import ConfigManager
from oboyu.indexer.config.indexer_config import IndexerConfig
from oboyu.mcp.context import db_path_global, mcp


class MCPService:
    """Service for handling MCP server operations."""
    
    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize the MCP service.
        
        Args:
            config_manager: Configuration manager instance

        """
        self.config_manager = config_manager
    
    def create_indexer_config(self, db_path: Optional[str] = None) -> IndexerConfig:
        """Create indexer configuration for MCP server.
        
        Args:
            db_path: Optional database path override
            
        Returns:
            IndexerConfig instance

        """
        indexer_config_dict = self.config_manager.get_section("indexer")
        
        # Handle database path with clear precedence
        resolved_db_path = self.config_manager.resolve_db_path(
            Path(db_path) if db_path else None,
            indexer_config_dict
        )
        indexer_config_dict["db_path"] = str(resolved_db_path)
        
        from oboyu.indexer.config.model_config import ModelConfig
        from oboyu.indexer.config.processing_config import ProcessingConfig
        from oboyu.indexer.config.search_config import SearchConfig
        
        # Create modular config from dict
        model_config = ModelConfig(
            use_reranker=indexer_config_dict.get("use_reranker", False)
        )
        search_config = SearchConfig(
            use_reranker=indexer_config_dict.get("use_reranker", False)
        )
        processing_config = ProcessingConfig(db_path=Path(indexer_config_dict["db_path"]))
        
        return IndexerConfig(
            model=model_config,
            search=search_config,
            processing=processing_config
        )
    
    def start_server(
        self,
        db_path: Optional[Path] = None,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        port: Optional[int] = None,
    ) -> None:
        """Start the MCP server.
        
        Args:
            db_path: Optional database path override
            transport: Transport mechanism
            port: Port number for HTTP transport
            
        Raises:
            Exception: If server startup fails

        """
        # Create indexer configuration to get resolved database path
        indexer_config = self.create_indexer_config(db_path=str(db_path) if db_path else None)
        resolved_db_path = str(indexer_config.processing.db_path if indexer_config.processing else "oboyu.db")
        
        # Store DB path in a global variable that our tools can access
        db_path_global.value = resolved_db_path
        
        # Run the server
        mcp.run(transport)
    
    def get_database_path(self, db_path: Optional[Path] = None) -> str:
        """Get the resolved database path.
        
        Args:
            db_path: Optional database path override
            
        Returns:
            Resolved database path as string

        """
        indexer_config = self.create_indexer_config(db_path=str(db_path) if db_path else None)
        return str(indexer_config.processing.db_path if indexer_config.processing else "oboyu.db")
