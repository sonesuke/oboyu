"""Configuration handling for Oboyu indexer.

This module provides utilities for loading and validating indexer configuration.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

# Default configuration values
DEFAULT_CONFIG = {
    "indexer": {
        # Document processing settings
        "chunk_size": 300,  # Default chunk size in characters (optimized for reranker compatibility)
        "chunk_overlap": 75,  # Default overlap between chunks (25% of chunk_size)
        # Embedding settings
        "embedding_model": "cl-nagoya/ruri-v3-30m",  # Default embedding model
        "embedding_device": "cpu",  # Default device for embeddings (cpu/cuda)
        "batch_size": 64,  # Default batch size for embedding generation
        "max_seq_length": 8192,  # Maximum sequence length (Ruri v3 default is 8192)
        "use_onnx": True,  # Whether to use ONNX optimization for faster inference
        # ONNX quantization settings
        "onnx_quantization": {
            "enabled": True,  # Whether to enable dynamic quantization (default: True)
            "method": "dynamic",  # Quantization method (dynamic, static, fp16)
            "weight_type": "uint8",  # Weight quantization type (uint8, int8)
        },
        # ONNX optimization settings
        "onnx_optimization_level": "extended",  # Graph optimization level: none, basic, extended, all
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
        "max_workers": 2,  # Maximum number of worker threads for parallel processing
        # Reranker settings
        "reranker_model": "cl-nagoya/ruri-reranker-small",  # Lightweight Japanese reranker
        # Alternative heavy model: "cl-nagoya/ruri-v3-reranker-310m"
        "use_reranker": False,  # Whether to use reranker for search results
        "reranker_use_onnx": False,  # Whether to use ONNX optimization for reranker (PyTorch is faster for most cases)
        "reranker_device": "cpu",  # Device for reranker (cpu/cuda)
        "reranker_top_k_multiplier": 2,  # Multiplier for initial retrieval (2x final top_k)
        "reranker_batch_size": 16,  # Batch size for reranking (reduced for better latency)
        "reranker_max_length": 512,  # Maximum sequence length for reranker
        "reranker_threshold": None,  # Minimum score threshold (None = no threshold)
        # BM25 settings
        "bm25_k1": 1.2,  # BM25 k1 parameter (term frequency saturation)
        "bm25_b": 0.75,  # BM25 b parameter (document length normalization)
        "bm25_min_token_length": 2,  # Minimum token length for BM25
        "use_japanese_tokenizer": True,  # Use Japanese morphological analyzer
        # Incremental indexing settings
        "incremental": {
            "enabled": True,  # Whether incremental indexing is enabled by default
            "change_detection_strategy": "smart",  # Default strategy: timestamp, hash, smart
            "hash_algorithm": "sha256",  # Hash algorithm for content comparison
            "track_file_metadata": True,  # Whether to track file metadata in database
            "cleanup_deleted_files": True,  # Whether to cleanup deleted files during incremental
            "batch_size_for_change_detection": 1000,  # Batch size for change detection
            "max_files_for_hash_checking": 10000,  # Max files before falling back to timestamp only
        },
    }
}

# Default values for individual settings, used for validation
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 75
DEFAULT_EMBEDDING_MODEL = "cl-nagoya/ruri-v3-30m"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_USE_ONNX = True
# ONNX quantization defaults
DEFAULT_ONNX_QUANTIZATION_ENABLED = True
DEFAULT_ONNX_QUANTIZATION_METHOD = "dynamic"
DEFAULT_ONNX_QUANTIZATION_WEIGHT_TYPE = "uint8"
DEFAULT_ONNX_OPTIMIZATION_LEVEL = "extended"
DEFAULT_DOCUMENT_PREFIX = "検索文書: "
DEFAULT_QUERY_PREFIX = "検索クエリ: "
DEFAULT_TOPIC_PREFIX = "トピック: "
DEFAULT_GENERAL_PREFIX = ""
# No default DB_PATH - must be explicitly provided
DEFAULT_EF_CONSTRUCTION = 128
DEFAULT_EF_SEARCH = 64
DEFAULT_M = 16
DEFAULT_M0 = None
DEFAULT_MAX_WORKERS = 2
# Reranker defaults
DEFAULT_RERANKER_MODEL = "cl-nagoya/ruri-reranker-small"
DEFAULT_USE_RERANKER = False
DEFAULT_RERANKER_USE_ONNX = True
DEFAULT_RERANKER_DEVICE = "cpu"
DEFAULT_RERANKER_TOP_K_MULTIPLIER = 2
DEFAULT_RERANKER_BATCH_SIZE = 64
DEFAULT_RERANKER_MAX_LENGTH = 512
DEFAULT_RERANKER_THRESHOLD = None
# BM25 defaults
DEFAULT_BM25_K1 = 1.2
DEFAULT_BM25_B = 0.75
DEFAULT_BM25_MIN_TOKEN_LENGTH = 2
DEFAULT_USE_JAPANESE_TOKENIZER = True
# Incremental indexing defaults
DEFAULT_INCREMENTAL_ENABLED = True
DEFAULT_CHANGE_DETECTION_STRATEGY = "smart"
DEFAULT_HASH_ALGORITHM = "sha256"
DEFAULT_TRACK_FILE_METADATA = True
DEFAULT_CLEANUP_DELETED_FILES = True
DEFAULT_BATCH_SIZE_FOR_CHANGE_DETECTION = 1000
DEFAULT_MAX_FILES_FOR_HASH_CHECKING = 10000


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

        # Validate basic settings
        self._validate_basic_settings(indexer_config)

        # Validate embedding settings
        self._validate_embedding_settings(indexer_config)

        # Validate ONNX quantization settings
        self._validate_onnx_quantization_settings(indexer_config)

        # Validate ONNX optimization settings
        self._validate_onnx_optimization_settings(indexer_config)

        # Validate prefix settings
        self._validate_prefix_settings(indexer_config)

        # Validate database settings
        self._validate_database_settings(indexer_config)

        # Validate VSS parameters
        self._validate_vss_settings(indexer_config)

        # Validate reranker settings
        self._validate_reranker_settings(indexer_config)
        
        # Validate chunk size compatibility with reranker
        self._validate_chunk_size_for_reranker(indexer_config)

        # Validate BM25 settings
        self._validate_bm25_settings(indexer_config)
        
        # Validate incremental indexing settings
        self._validate_incremental_settings(indexer_config)

    def _validate_basic_settings(self, indexer_config: Dict[str, Any]) -> None:
        """Validate basic indexer settings."""
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

        # Validate batch_size - must be a positive integer
        if not isinstance(indexer_config.get("batch_size"), int) or indexer_config.get("batch_size", 0) <= 0:
            indexer_config["batch_size"] = DEFAULT_BATCH_SIZE

        # Validate max_seq_length - must be a positive integer
        if not isinstance(indexer_config.get("max_seq_length"), int) or indexer_config.get("max_seq_length", 0) <= 0:
            indexer_config["max_seq_length"] = DEFAULT_MAX_SEQ_LENGTH

        # Validate max_workers - must be a positive integer
        if not isinstance(indexer_config.get("max_workers"), int) or indexer_config.get("max_workers", 0) <= 0:
            indexer_config["max_workers"] = DEFAULT_MAX_WORKERS

    def _validate_embedding_settings(self, indexer_config: Dict[str, Any]) -> None:
        """Validate embedding-related settings."""
        # Validate embedding_model - must be a non-empty string
        if not isinstance(indexer_config.get("embedding_model"), str) or not indexer_config.get("embedding_model"):
            indexer_config["embedding_model"] = DEFAULT_EMBEDDING_MODEL

        # Validate embedding_device - must be 'cpu' or 'cuda'
        if not isinstance(indexer_config.get("embedding_device"), str) or indexer_config.get("embedding_device") not in ["cpu", "cuda"]:
            indexer_config["embedding_device"] = DEFAULT_EMBEDDING_DEVICE

        # Validate use_onnx - must be a boolean
        if not isinstance(indexer_config.get("use_onnx"), bool):
            indexer_config["use_onnx"] = DEFAULT_USE_ONNX

    def _validate_onnx_quantization_settings(self, indexer_config: Dict[str, Any]) -> None:
        """Validate ONNX quantization settings."""
        # Ensure onnx_quantization dict exists
        if not isinstance(indexer_config.get("onnx_quantization"), dict):
            indexer_config["onnx_quantization"] = {}

        onnx_quant = indexer_config["onnx_quantization"]

        # Validate enabled - must be a boolean
        if not isinstance(onnx_quant.get("enabled"), bool):
            onnx_quant["enabled"] = DEFAULT_ONNX_QUANTIZATION_ENABLED

        # Validate method - must be one of the supported methods
        if onnx_quant.get("method") not in ["dynamic", "static", "fp16"]:
            onnx_quant["method"] = DEFAULT_ONNX_QUANTIZATION_METHOD

        # Validate weight_type - must be one of the supported types
        if onnx_quant.get("weight_type") not in ["uint8", "int8"]:
            onnx_quant["weight_type"] = DEFAULT_ONNX_QUANTIZATION_WEIGHT_TYPE
    
    def _validate_onnx_optimization_settings(self, indexer_config: Dict[str, Any]) -> None:
        """Validate ONNX optimization settings."""
        # Validate onnx_optimization_level - must be one of the supported levels
        if indexer_config.get("onnx_optimization_level") not in ["none", "basic", "extended", "all"]:
            indexer_config["onnx_optimization_level"] = DEFAULT_ONNX_OPTIMIZATION_LEVEL

    def _validate_prefix_settings(self, indexer_config: Dict[str, Any]) -> None:
        """Validate prefix settings."""
        # Validate prefixes - must be strings
        if not isinstance(indexer_config.get("document_prefix"), str):
            indexer_config["document_prefix"] = DEFAULT_DOCUMENT_PREFIX
        if not isinstance(indexer_config.get("query_prefix"), str):
            indexer_config["query_prefix"] = DEFAULT_QUERY_PREFIX
        if not isinstance(indexer_config.get("topic_prefix"), str):
            indexer_config["topic_prefix"] = DEFAULT_TOPIC_PREFIX
        if not isinstance(indexer_config.get("general_prefix"), str):
            indexer_config["general_prefix"] = DEFAULT_GENERAL_PREFIX

    def _validate_database_settings(self, indexer_config: Dict[str, Any]) -> None:
        """Validate database settings."""
        # Validate db_path - must be a non-empty string and must be provided
        if not isinstance(indexer_config.get("db_path"), str) or not indexer_config.get("db_path"):
            raise ValueError("Database path (db_path) must be provided and cannot be empty")

    def _validate_vss_settings(self, indexer_config: Dict[str, Any]) -> None:
        """Validate VSS parameters."""
        # Validate VSS parameters - must be positive integers
        if not isinstance(indexer_config.get("ef_construction"), int) or indexer_config.get("ef_construction", 0) <= 0:
            indexer_config["ef_construction"] = DEFAULT_EF_CONSTRUCTION
        if not isinstance(indexer_config.get("ef_search"), int) or indexer_config.get("ef_search", 0) <= 0:
            indexer_config["ef_search"] = DEFAULT_EF_SEARCH
        if not isinstance(indexer_config.get("m"), int) or indexer_config.get("m", 0) <= 0:
            indexer_config["m"] = DEFAULT_M

        # m0 can be None (meaning 2*M) or a positive integer
        if indexer_config.get("m0") is not None and (not isinstance(indexer_config.get("m0"), int) or indexer_config.get("m0", 0) <= 0):
            indexer_config["m0"] = DEFAULT_M0

    def _validate_reranker_settings(self, indexer_config: Dict[str, Any]) -> None:
        """Validate reranker settings."""
        # Validate reranker_model - must be a non-empty string
        if not isinstance(indexer_config.get("reranker_model"), str) or not indexer_config.get("reranker_model"):
            indexer_config["reranker_model"] = DEFAULT_RERANKER_MODEL

        # Validate use_reranker - must be a boolean
        if not isinstance(indexer_config.get("use_reranker"), bool):
            indexer_config["use_reranker"] = DEFAULT_USE_RERANKER

        # Validate reranker_use_onnx - must be a boolean
        if not isinstance(indexer_config.get("reranker_use_onnx"), bool):
            indexer_config["reranker_use_onnx"] = DEFAULT_RERANKER_USE_ONNX

        # Validate reranker_device - must be 'cpu' or 'cuda'
        if indexer_config.get("reranker_device") not in ["cpu", "cuda"]:
            indexer_config["reranker_device"] = DEFAULT_RERANKER_DEVICE

        # Validate reranker_top_k_multiplier - must be a positive integer
        if not isinstance(indexer_config.get("reranker_top_k_multiplier"), int) or indexer_config.get("reranker_top_k_multiplier", 0) <= 0:
            indexer_config["reranker_top_k_multiplier"] = DEFAULT_RERANKER_TOP_K_MULTIPLIER

        # Validate reranker_batch_size - must be a positive integer
        if not isinstance(indexer_config.get("reranker_batch_size"), int) or indexer_config.get("reranker_batch_size", 0) <= 0:
            indexer_config["reranker_batch_size"] = DEFAULT_RERANKER_BATCH_SIZE

        # Validate reranker_max_length - must be a positive integer
        if not isinstance(indexer_config.get("reranker_max_length"), int) or indexer_config.get("reranker_max_length", 0) <= 0:
            indexer_config["reranker_max_length"] = DEFAULT_RERANKER_MAX_LENGTH

        # Validate reranker_threshold - must be None or a float between 0 and 1
        threshold = indexer_config.get("reranker_threshold")
        if threshold is not None and (not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1):
            indexer_config["reranker_threshold"] = DEFAULT_RERANKER_THRESHOLD

    def _validate_chunk_size_for_reranker(self, indexer_config: Dict[str, Any]) -> None:
        """Validate that chunk size is compatible with reranker constraints."""
        import logging
        
        chunk_size = indexer_config.get("chunk_size", DEFAULT_CHUNK_SIZE)
        reranker_max_length = indexer_config.get("reranker_max_length", DEFAULT_RERANKER_MAX_LENGTH)
        
        # Conservative estimate: ~60% of reranker max length should be available for document content
        # This accounts for query text, prefixes, and separator tokens
        max_recommended_chunk_size = int(reranker_max_length * 0.6)
        
        if chunk_size > max_recommended_chunk_size:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Chunk size ({chunk_size}) may exceed reranker capacity "
                f"(recommended: <{max_recommended_chunk_size} for reranker_max_length={reranker_max_length}). "
                f"This may cause truncation warnings and degraded reranking quality. "
                f"Consider reducing chunk_size."
            )

    def _validate_bm25_settings(self, indexer_config: dict[str, Any]) -> None:
        """Validate BM25 settings."""
        # Validate bm25_k1 - must be a positive float
        if not isinstance(indexer_config.get("bm25_k1"), (int, float)) or indexer_config.get("bm25_k1", 0) <= 0:
            indexer_config["bm25_k1"] = DEFAULT_BM25_K1

        # Validate bm25_b - must be a float between 0 and 1
        if not isinstance(indexer_config.get("bm25_b"), (int, float)) or indexer_config.get("bm25_b", -1) < 0 or indexer_config.get("bm25_b", 2) > 1:
            indexer_config["bm25_b"] = DEFAULT_BM25_B

        # Validate bm25_min_token_length - must be a positive integer
        if not isinstance(indexer_config.get("bm25_min_token_length"), int) or indexer_config.get("bm25_min_token_length", 0) <= 0:
            indexer_config["bm25_min_token_length"] = DEFAULT_BM25_MIN_TOKEN_LENGTH

        # Validate use_japanese_tokenizer - must be a boolean
        if not isinstance(indexer_config.get("use_japanese_tokenizer"), bool):
            indexer_config["use_japanese_tokenizer"] = DEFAULT_USE_JAPANESE_TOKENIZER
    
    def _validate_incremental_settings(self, indexer_config: Dict[str, Any]) -> None:
        """Validate incremental indexing settings."""
        # Ensure incremental dict exists
        if "incremental" not in indexer_config:
            indexer_config["incremental"] = {}
        
        incremental_config = indexer_config["incremental"]
        
        # Validate enabled - must be a boolean
        if not isinstance(incremental_config.get("enabled"), bool):
            incremental_config["enabled"] = DEFAULT_INCREMENTAL_ENABLED
        
        # Validate change_detection_strategy - must be one of allowed values
        valid_strategies = ["timestamp", "hash", "smart"]
        if incremental_config.get("change_detection_strategy") not in valid_strategies:
            incremental_config["change_detection_strategy"] = DEFAULT_CHANGE_DETECTION_STRATEGY
        
        # Validate hash_algorithm - must be a valid hash algorithm
        valid_algorithms = ["sha256", "sha512", "md5"]
        if incremental_config.get("hash_algorithm") not in valid_algorithms:
            incremental_config["hash_algorithm"] = DEFAULT_HASH_ALGORITHM
        
        # Validate track_file_metadata - must be a boolean
        if not isinstance(incremental_config.get("track_file_metadata"), bool):
            incremental_config["track_file_metadata"] = DEFAULT_TRACK_FILE_METADATA
        
        # Validate cleanup_deleted_files - must be a boolean
        if not isinstance(incremental_config.get("cleanup_deleted_files"), bool):
            incremental_config["cleanup_deleted_files"] = DEFAULT_CLEANUP_DELETED_FILES
        
        # Validate batch_size_for_change_detection - must be a positive integer
        if not isinstance(incremental_config.get("batch_size_for_change_detection"), int) or incremental_config.get("batch_size_for_change_detection", 0) <= 0:
            incremental_config["batch_size_for_change_detection"] = DEFAULT_BATCH_SIZE_FOR_CHANGE_DETECTION
        
        # Validate max_files_for_hash_checking - must be a positive integer
        if not isinstance(incremental_config.get("max_files_for_hash_checking"), int) or incremental_config.get("max_files_for_hash_checking", 0) <= 0:
            incremental_config["max_files_for_hash_checking"] = DEFAULT_MAX_FILES_FOR_HASH_CHECKING

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
    def use_onnx(self) -> bool:
        """Whether to use ONNX optimization for faster inference."""
        return bool(self.config["indexer"]["use_onnx"])

    @property
    def onnx_quantization_enabled(self) -> bool:
        """Whether ONNX quantization is enabled."""
        return bool(self.config["indexer"]["onnx_quantization"]["enabled"])

    @property
    def onnx_quantization_method(self) -> str:
        """ONNX quantization method."""
        return str(self.config["indexer"]["onnx_quantization"]["method"])

    @property
    def onnx_quantization_weight_type(self) -> str:
        """ONNX quantization weight type."""
        return str(self.config["indexer"]["onnx_quantization"]["weight_type"])

    @property
    def onnx_quantization_config(self) -> Dict[str, Any]:
        """Full ONNX quantization configuration."""
        return dict(self.config["indexer"]["onnx_quantization"])
    
    @property
    def onnx_optimization_level(self) -> str:
        """ONNX graph optimization level."""
        return str(self.config["indexer"]["onnx_optimization_level"])

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

    @property
    def reranker_model(self) -> str:
        """Reranker model name."""
        return str(self.config["indexer"]["reranker_model"])

    @property
    def use_reranker(self) -> bool:
        """Whether to use reranker for search results."""
        return bool(self.config["indexer"]["use_reranker"])

    @property
    def reranker_use_onnx(self) -> bool:
        """Whether to use ONNX optimization for reranker."""
        return bool(self.config["indexer"]["reranker_use_onnx"])

    @property
    def reranker_device(self) -> str:
        """Device for reranker (cpu/cuda)."""
        return str(self.config["indexer"]["reranker_device"])

    @property
    def reranker_top_k_multiplier(self) -> int:
        """Multiplier for initial retrieval."""
        return int(self.config["indexer"]["reranker_top_k_multiplier"])

    @property
    def reranker_batch_size(self) -> int:
        """Batch size for reranking."""
        return int(self.config["indexer"]["reranker_batch_size"])

    @property
    def reranker_max_length(self) -> int:
        """Maximum sequence length for reranker."""
        return int(self.config["indexer"]["reranker_max_length"])

    @property
    def reranker_threshold(self) -> Optional[float]:
        """Minimum score threshold for reranker."""
        threshold = self.config["indexer"]["reranker_threshold"]
        return float(threshold) if threshold is not None else None

    @property
    def bm25_k1(self) -> float:
        """BM25 k1 parameter."""
        return float(self.config["indexer"]["bm25_k1"])

    @property
    def bm25_b(self) -> float:
        """BM25 b parameter."""
        return float(self.config["indexer"]["bm25_b"])

    @property
    def bm25_min_token_length(self) -> int:
        """Minimum token length for BM25."""
        return int(self.config["indexer"]["bm25_min_token_length"])

    @property
    def use_japanese_tokenizer(self) -> bool:
        """Whether to use Japanese tokenizer."""
        return bool(self.config["indexer"]["use_japanese_tokenizer"])
    
    # Incremental indexing properties
    @property
    def incremental_enabled(self) -> bool:
        """Whether incremental indexing is enabled."""
        return bool(self.config["indexer"]["incremental"]["enabled"])
    
    @property
    def change_detection_strategy(self) -> str:
        """Strategy for detecting file changes."""
        return str(self.config["indexer"]["incremental"]["change_detection_strategy"])
    
    @property
    def hash_algorithm(self) -> str:
        """Hash algorithm for content comparison."""
        return str(self.config["indexer"]["incremental"]["hash_algorithm"])
    
    @property
    def track_file_metadata(self) -> bool:
        """Whether to track file metadata in database."""
        return bool(self.config["indexer"]["incremental"]["track_file_metadata"])
    
    @property
    def cleanup_deleted_files(self) -> bool:
        """Whether to cleanup deleted files during incremental indexing."""
        return bool(self.config["indexer"]["incremental"]["cleanup_deleted_files"])
    
    @property
    def batch_size_for_change_detection(self) -> int:
        """Batch size for change detection processing."""
        return int(self.config["indexer"]["incremental"]["batch_size_for_change_detection"])
    
    @property
    def max_files_for_hash_checking(self) -> int:
        """Maximum files before falling back to timestamp-only checking."""
        return int(self.config["indexer"]["incremental"]["max_files_for_hash_checking"])
    
    @property
    def incremental_config(self) -> Dict[str, Any]:
        """Full incremental indexing configuration."""
        return dict(self.config["indexer"]["incremental"])


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
