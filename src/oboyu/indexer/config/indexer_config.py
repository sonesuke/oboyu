"""Main indexer configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from oboyu.indexer.config.model_config import ModelConfig
from oboyu.indexer.config.processing_config import ProcessingConfig
from oboyu.indexer.config.search_config import SearchConfig


@dataclass
class IndexerConfig:
    """Main indexer configuration combining all sub-configurations."""
    
    model: ModelConfig = None
    search: SearchConfig = None
    processing: ProcessingConfig = None
    
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
        return self.processing.db_path
    
    @db_path.setter
    def db_path(self, value: Union[str, Path]) -> None:
        """Set database path."""
        self.processing.db_path = Path(value)
    
    @property
    def chunk_size(self) -> int:
        """Chunk size."""
        return self.processing.chunk_size
    
    @property
    def chunk_overlap(self) -> int:
        """Chunk overlap."""
        return self.processing.chunk_overlap
    
    @property
    def embedding_model(self) -> str:
        """Embedding model name."""
        return self.model.embedding_model
    
    @property
    def embedding_device(self) -> str:
        """Embedding device."""
        return self.model.embedding_device
    
    @property
    def use_reranker(self) -> bool:
        """Whether to use reranker."""
        return self.search.use_reranker or self.model.use_reranker
    
    @property
    def reranker_model(self) -> str:
        """Reranker model name."""
        return self.model.reranker_model
