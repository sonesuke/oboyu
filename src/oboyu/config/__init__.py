"""Unified configuration system for oboyu."""

# Import base configuration management
from oboyu.config.base import ConfigManager

# Import configuration schemas  
from oboyu.config.schema import (
    ConfigSchema,
    CrawlerConfigSchema,
    IndexerConfigSchema,
    QueryConfigSchema,
)

# Import component-specific configurations
from oboyu.config.crawler import CrawlerConfig, DEFAULT_CONFIG as CRAWLER_DEFAULTS
from oboyu.config.indexer import (
    IndexerConfig,
    ModelConfig,
    SearchConfig,
    ProcessingConfig,
    DEFAULT_CONFIG as INDEXER_DEFAULTS,
)
from oboyu.config.query import QueryConfig, DEFAULT_CONFIG as QUERY_DEFAULTS

# Legacy imports for backward compatibility
try:
    from oboyu.common.config import ConfigManager as LegacyConfigManager
    from oboyu.common.config_schema import (
        ConfigSchema as LegacyConfigSchema,
        CrawlerConfigSchema as LegacyCrawlerConfigSchema,
        IndexerConfigSchema as LegacyIndexerConfigSchema,
        QueryConfigSchema as LegacyQueryConfigSchema,
    )
except ImportError:
    # If old config doesn't exist, use new ones
    LegacyConfigManager = ConfigManager
    LegacyConfigSchema = ConfigSchema
    LegacyCrawlerConfigSchema = CrawlerConfigSchema  
    LegacyIndexerConfigSchema = IndexerConfigSchema
    LegacyQueryConfigSchema = QueryConfigSchema

# Unified defaults
UNIFIED_DEFAULTS = {
    "crawler": CRAWLER_DEFAULTS["crawler"],
    "indexer": INDEXER_DEFAULTS["indexer"],
    "query": QUERY_DEFAULTS["query"],
}

__all__ = [
    # Configuration manager
    "ConfigManager",
    
    # Configuration schemas
    "ConfigSchema",
    "CrawlerConfigSchema", 
    "IndexerConfigSchema",
    "QueryConfigSchema",
    
    # Component configurations
    "CrawlerConfig",
    "IndexerConfig",
    "ModelConfig",
    "SearchConfig", 
    "ProcessingConfig",
    "QueryConfig",
    
    # Defaults
    "CRAWLER_DEFAULTS",
    "INDEXER_DEFAULTS", 
    "QUERY_DEFAULTS",
    "UNIFIED_DEFAULTS",
    
    # Legacy compatibility
    "LegacyConfigManager",
    "LegacyConfigSchema",
    "LegacyCrawlerConfigSchema",
    "LegacyIndexerConfigSchema",
    "LegacyQueryConfigSchema",
]