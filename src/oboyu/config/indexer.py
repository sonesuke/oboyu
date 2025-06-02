"""Consolidated indexer configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union


@dataclass
class ModelConfig:
    """Model-related configuration."""

    # Embedding model settings
    embedding_model: str = "cl-nagoya/ruri-v3-30m"
    embedding_device: str = "cpu"
    batch_size: int = 64
    max_seq_length: int = 8192
    use_onnx: bool = True

    # ONNX optimization settings
    onnx_quantization: Optional[Dict[str, object]] = None
    onnx_optimization_level: str = "extended"

    # Prefix scheme settings (Ruri v3's 1+3 prefix scheme)
    document_prefix: str = "検索文書: "
    query_prefix: str = "検索クエリ: "
    topic_prefix: str = "トピック: "
    general_prefix: str = ""

    # Reranker settings
    reranker_model: str = "cl-nagoya/ruri-reranker-small"
    use_reranker: bool = False
    reranker_use_onnx: bool = False
    reranker_device: str = "cpu"
    reranker_batch_size: int = 16
    reranker_max_length: int = 512

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        if self.onnx_quantization is None:
            self.onnx_quantization = {
                "enabled": True,
                "method": "dynamic",
                "weight_type": "uint8",
            }


@dataclass
class SearchConfig:
    """Search-related configuration."""

    # RRF (Reciprocal Rank Fusion) parameter
    rrf_k: int = 60

    # Search parameters
    use_reranker: bool = False
    top_k_multiplier: int = 2

    # BM25 parameters
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_min_token_length: int = 2

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        # Validate RRF k parameter
        if self.rrf_k <= 0:
            self.rrf_k = 60


@dataclass
class ProcessingConfig:
    """Processing-related configuration."""

    # Database settings
    db_path: Path = Path("oboyu_database.db")
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Processing settings
    batch_size: int = 64
    show_progress: bool = True
    
    # Validation settings
    min_doc_length: int = 50


@dataclass
class IndexerConfig:
    """Main indexer configuration combining all sub-configurations."""

    model: Optional[ModelConfig] = None
    search: Optional[SearchConfig] = None
    processing: Optional[ProcessingConfig] = None

    def __post_init__(self) -> None:
        """Post-initialization to create default sub-configs if not provided."""
        if self.model is None:
            self.model = ModelConfig()
        if self.search is None:
            self.search = SearchConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()

    # Convenience properties for backward compatibility
    @property
    def db_path(self) -> Path:
        """Database path."""
        assert self.processing is not None, "ProcessingConfig should be initialized"
        return Path(self.processing.db_path)

    @db_path.setter
    def db_path(self, value: Union[str, Path]) -> None:
        """Set database path."""
        assert self.processing is not None, "ProcessingConfig should be initialized"
        self.processing.db_path = Path(value)

    @property
    def chunk_size(self) -> int:
        """Chunk size."""
        assert self.processing is not None, "ProcessingConfig should be initialized"
        return self.processing.chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Chunk overlap."""
        assert self.processing is not None, "ProcessingConfig should be initialized"
        return self.processing.chunk_overlap

    @property
    def embedding_model(self) -> str:
        """Embedding model name."""
        assert self.model is not None, "ModelConfig should be initialized"
        return self.model.embedding_model

    @property
    def embedding_device(self) -> str:
        """Embedding device."""
        assert self.model is not None, "ModelConfig should be initialized"
        return self.model.embedding_device

    @property
    def use_reranker(self) -> bool:
        """Whether to use reranker."""
        assert self.search is not None and self.model is not None, "Configs should be initialized"
        return self.search.use_reranker or self.model.use_reranker

    @property
    def reranker_model(self) -> str:
        """Reranker model name."""
        assert self.model is not None, "ModelConfig should be initialized"
        return self.model.reranker_model


# Default configuration instance
DEFAULT_CONFIG = {
    "indexer": {
        "model": ModelConfig(),
        "search": SearchConfig(),
        "processing": ProcessingConfig(),
    }
}
