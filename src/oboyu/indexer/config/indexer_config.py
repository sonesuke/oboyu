"""Main indexer configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from oboyu.indexer.config.model_config import ModelConfig
from oboyu.indexer.config.processing_config import ProcessingConfig
from oboyu.indexer.config.search_config import SearchConfig


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
        """Embedding device (CPU only)."""
        return "cpu"

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
