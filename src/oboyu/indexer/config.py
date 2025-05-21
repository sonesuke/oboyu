"""Configuration handling for Oboyu indexer.

This module provides utilities for loading and validating indexer configuration.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

# Default configuration values
DEFAULT_CONFIG = {
    "indexer": {
        # Document processing settings
        "chunk_size": 1024,  # Default chunk size in characters
        "chunk_overlap": 256,  # Default overlap between chunks

        # Embedding settings
        "embedding_model": "cl-nagoya/ruri-v3-30m",  # Default embedding model
        "embedding_device": "cpu",  # Default device for embeddings (cpu/cuda)
        "batch_size": 128,  # Default batch size for embedding generation
        "max_seq_length": 8192,  # Maximum sequence length (Ruri v3 default is 8192)

        # Prefix scheme settings (Ruri v3's 1+3 prefix scheme)
        "document_prefix": "検索文書: ",  # Prefix for documents to be indexed
        "query_prefix": "検索クエリ: ",  # Prefix for search queries
        "topic_prefix": "トピック: ",  # Prefix for topic information
        "general_prefix": "",  # Prefix for general semantic encoding

        # Database settings
        # db_path must be explicitly provided by the caller

        # VSS (Vector Similarity Search) settings
        "ef_construction": 128,  # Index construction parameter (build-time)
        "ef_search": 64,  # Search time parameter (quality vs. speed)
        "m": 16,  # Number of bidirectional links in HNSW graph
        "m0": None,  # Level-0 connections (None means use 2*M)

        # Processing settings
        "max_workers": 4,  # Maximum number of worker threads for parallel processing
    }
}

# Default values for individual settings, used for validation
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 256
DEFAULT_EMBEDDING_MODEL = "cl-nagoya/ruri-v3-30m"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_DOCUMENT_PREFIX = "検索文書: "
DEFAULT_QUERY_PREFIX = "検索クエリ: "
DEFAULT_TOPIC_PREFIX = "トピック: "
DEFAULT_GENERAL_PREFIX = ""
# No default DB_PATH - must be explicitly provided
DEFAULT_EF_CONSTRUCTION = 128
DEFAULT_EF_SEARCH = 64
DEFAULT_M = 16
DEFAULT_M0 = None
DEFAULT_MAX_WORKERS = 4


class IndexerConfig:
    """Configuration handler for the indexer."""

    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize indexer configuration.

        Args:
            config_dict: Optional configuration dictionary
            config_path: Optional path to configuration file

        Note:
            If both config_dict and config_path are provided, config_dict takes precedence.
            If neither is provided, default configuration is used.

        """
        self.config: Dict[str, Any] = {}

        # Start with default configuration
        self.config.update(DEFAULT_CONFIG)

        # If config path is provided, load from file
        if config_path:
            self._load_from_file(config_path)

        # If config dict is provided, override with it
        if config_dict and "indexer" in config_dict:
            self.config["indexer"].update(config_dict["indexer"])

        # Validate the configuration
        self._validate()

    def _load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from a file.

        Args:
            config_path: Path to configuration file

        Note:
            This is a placeholder. In a real implementation, this would use
            a YAML or JSON parser to load the configuration.

        """
        # In a real implementation, this would parse a YAML or JSON file
        # For example:
        # import yaml
        # with open(config_path, "r") as f:
        #     loaded_config = yaml.safe_load(f)
        #     if loaded_config and "indexer" in loaded_config:
        #         self.config["indexer"].update(loaded_config["indexer"])
        pass

    def _validate(self) -> None:
        """Validate the configuration values."""
        # Get the indexer config dict
        indexer_config = self.config["indexer"]

        # Validate chunk_size - must be a positive integer
        if not isinstance(indexer_config.get("chunk_size"), int) or indexer_config.get("chunk_size", 0) <= 0:
            indexer_config["chunk_size"] = DEFAULT_CHUNK_SIZE

        # Validate chunk_overlap - must be a non-negative integer less than chunk_size
        if (
            not isinstance(indexer_config.get("chunk_overlap"), int)
            or indexer_config.get("chunk_overlap", 0) < 0
            or indexer_config.get("chunk_overlap", 0) >= indexer_config.get("chunk_size", DEFAULT_CHUNK_SIZE)
        ):
            indexer_config["chunk_overlap"] = DEFAULT_CHUNK_OVERLAP

        # Validate embedding_model - must be a non-empty string
        if not isinstance(indexer_config.get("embedding_model"), str) or not indexer_config.get("embedding_model"):
            indexer_config["embedding_model"] = DEFAULT_EMBEDDING_MODEL

        # Validate embedding_device - must be 'cpu' or 'cuda'
        if (
            not isinstance(indexer_config.get("embedding_device"), str)
            or indexer_config.get("embedding_device") not in ["cpu", "cuda"]
        ):
            indexer_config["embedding_device"] = DEFAULT_EMBEDDING_DEVICE

        # Validate batch_size - must be a positive integer
        if not isinstance(indexer_config.get("batch_size"), int) or indexer_config.get("batch_size", 0) <= 0:
            indexer_config["batch_size"] = DEFAULT_BATCH_SIZE

        # Validate max_seq_length - must be a positive integer
        if not isinstance(indexer_config.get("max_seq_length"), int) or indexer_config.get("max_seq_length", 0) <= 0:
            indexer_config["max_seq_length"] = DEFAULT_MAX_SEQ_LENGTH

        # Validate prefixes - must be strings
        if not isinstance(indexer_config.get("document_prefix"), str):
            indexer_config["document_prefix"] = DEFAULT_DOCUMENT_PREFIX
        if not isinstance(indexer_config.get("query_prefix"), str):
            indexer_config["query_prefix"] = DEFAULT_QUERY_PREFIX
        if not isinstance(indexer_config.get("topic_prefix"), str):
            indexer_config["topic_prefix"] = DEFAULT_TOPIC_PREFIX
        if not isinstance(indexer_config.get("general_prefix"), str):
            indexer_config["general_prefix"] = DEFAULT_GENERAL_PREFIX

        # Validate db_path - must be a non-empty string and must be provided
        if not isinstance(indexer_config.get("db_path"), str) or not indexer_config.get("db_path"):
            raise ValueError("Database path (db_path) must be provided and cannot be empty")

        # Validate VSS parameters - must be positive integers
        if not isinstance(indexer_config.get("ef_construction"), int) or indexer_config.get("ef_construction", 0) <= 0:
            indexer_config["ef_construction"] = DEFAULT_EF_CONSTRUCTION
        if not isinstance(indexer_config.get("ef_search"), int) or indexer_config.get("ef_search", 0) <= 0:
            indexer_config["ef_search"] = DEFAULT_EF_SEARCH
        if not isinstance(indexer_config.get("m"), int) or indexer_config.get("m", 0) <= 0:
            indexer_config["m"] = DEFAULT_M

        # m0 can be None (meaning 2*M) or a positive integer
        if indexer_config.get("m0") is not None and (
            not isinstance(indexer_config.get("m0"), int) or indexer_config.get("m0", 0) <= 0
        ):
            indexer_config["m0"] = DEFAULT_M0

        # Validate max_workers - must be a positive integer
        if not isinstance(indexer_config.get("max_workers"), int) or indexer_config.get("max_workers", 0) <= 0:
            indexer_config["max_workers"] = DEFAULT_MAX_WORKERS

    @property
    def chunk_size(self) -> int:
        """Maximum chunk size in characters."""
        return int(self.config["indexer"]["chunk_size"])

    @property
    def chunk_overlap(self) -> int:
        """Chunk overlap in characters."""
        return int(self.config["indexer"]["chunk_overlap"])

    @property
    def embedding_model(self) -> str:
        """Embedding model name."""
        return str(self.config["indexer"]["embedding_model"])

    @property
    def embedding_device(self) -> str:
        """Device to use for embeddings (cpu/cuda)."""
        return str(self.config["indexer"]["embedding_device"])

    @property
    def batch_size(self) -> int:
        """Batch size for embedding generation."""
        return int(self.config["indexer"]["batch_size"])

    @property
    def max_seq_length(self) -> int:
        """Maximum sequence length for embedding model."""
        return int(self.config["indexer"]["max_seq_length"])

    @property
    def document_prefix(self) -> str:
        """Prefix for document texts."""
        return str(self.config["indexer"]["document_prefix"])

    @property
    def query_prefix(self) -> str:
        """Prefix for search queries."""
        return str(self.config["indexer"]["query_prefix"])

    @property
    def topic_prefix(self) -> str:
        """Prefix for topic information."""
        return str(self.config["indexer"]["topic_prefix"])

    @property
    def general_prefix(self) -> str:
        """Prefix for general semantic encoding."""
        return str(self.config["indexer"]["general_prefix"])

    @property
    def db_path(self) -> str:
        """Database file path."""
        return str(self.config["indexer"]["db_path"])

    @property
    def ef_construction(self) -> int:
        """HNSW index construction parameter."""
        return int(self.config["indexer"]["ef_construction"])

    @property
    def ef_search(self) -> int:
        """HNSW search time parameter."""
        return int(self.config["indexer"]["ef_search"])

    @property
    def m(self) -> int:
        """Number of bidirectional links in HNSW graph."""
        return int(self.config["indexer"]["m"])

    @property
    def m0(self) -> Optional[int]:
        """Level-0 connections in HNSW graph."""
        m0 = self.config["indexer"]["m0"]
        return int(m0) if m0 is not None else None

    @property
    def max_workers(self) -> int:
        """Maximum number of worker threads for parallel processing."""
        return int(self.config["indexer"]["max_workers"])


def load_default_config(db_path: str) -> IndexerConfig:
    """Load the default indexer configuration with the specified database path.

    Args:
        db_path: Database path to use

    Returns:
        Default indexer configuration with specified database path

    """
    return IndexerConfig(config_dict={"indexer": {"db_path": db_path}})


def load_config_from_file(config_path: Union[str, Path], db_path: Optional[str] = None) -> IndexerConfig:
    """Load indexer configuration from a file.

    Args:
        config_path: Path to configuration file
        db_path: Database path to use if not specified in the config file

    Returns:
        Indexer configuration loaded from file

    Raises:
        ValueError: If db_path is not provided and not in the config file

    """
    config = IndexerConfig(config_path=config_path)

    # If db_path was not provided in the config file and not passed as parameter,
    # this will raise a ValueError. This is handled in the IndexerConfig validation.

    return config
