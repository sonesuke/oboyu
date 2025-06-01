"""Embedding generation service."""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from oboyu.common.model_manager import EmbeddingModelManager
from oboyu.common.paths import EMBEDDING_CACHE_DIR, EMBEDDING_MODELS_DIR

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for document embeddings to avoid regeneration."""

    def __init__(self, cache_dir: Union[str, Path] = EMBEDDING_CACHE_DIR) -> None:
        """Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings

        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def get(self, text: str, model_name: str) -> Optional[NDArray[np.float32]]:
        """Retrieve embedding from cache.

        Args:
            text: Text to get embedding for
            model_name: Name of the embedding model

        Returns:
            Cached embedding array or None if not found

        """
        cache_key = self._get_cache_key(text, model_name)
        cache_file = self.cache_dir / f"{cache_key}.npy"

        if cache_file.exists():
            try:
                loaded_array = np.load(cache_file)
                # Cast to proper type annotation
                return loaded_array.astype(np.float32)  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)

        return None

    def set(self, text: str, model_name: str, embedding: NDArray[np.float32]) -> None:
        """Store embedding in cache.

        Args:
            text: Text the embedding was generated for
            model_name: Name of the embedding model
            embedding: Embedding array to cache

        """
        cache_key = self._get_cache_key(text, model_name)
        cache_file = self.cache_dir / f"{cache_key}.npy"

        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model.

        Args:
            text: Text content
            model_name: Model name

        Returns:
            Hash key for caching

        """
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()


class EmbeddingService:
    """Service for generating embeddings using various models."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-30m",
        device: str = "cpu",
        batch_size: int = 64,
        max_seq_length: int = 8192,
        query_prefix: str = "検索クエリ: ",
        use_onnx: bool = True,
        use_cache: bool = True,
        onnx_quantization_config: Optional[Dict[str, object]] = None,
        onnx_optimization_level: str = "extended",
    ) -> None:
        """Initialize embedding service.

        Args:
            model_name: Name of the embedding model
            device: Device to run model on (cpu/cuda)
            batch_size: Batch size for embedding generation
            max_seq_length: Maximum sequence length
            query_prefix: Prefix for search queries
            use_onnx: Whether to use ONNX optimization
            use_cache: Whether to use embedding cache
            onnx_quantization_config: ONNX quantization configuration
            onnx_optimization_level: ONNX optimization level

        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.query_prefix = query_prefix
        self.use_onnx = use_onnx
        self.use_cache = use_cache

        # Initialize model manager
        self.model_manager = EmbeddingModelManager(
            model_name=model_name,
            models_dir=EMBEDDING_MODELS_DIR,
            device=device,
            max_seq_length=max_seq_length,
            use_onnx=use_onnx,
            quantization_config=onnx_quantization_config,
            optimization_level=onnx_optimization_level,
        )

        # Initialize cache if enabled
        self.cache = EmbeddingCache() if use_cache else None

    @property
    def dimensions(self) -> Optional[int]:
        """Get embedding dimensions."""
        return self.model_manager.get_dimensions()

    def generate_embeddings(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding arrays

        """
        if not texts:
            return []

        # Use dictionary to store results by index to avoid None placeholders
        result_embeddings: Dict[int, NDArray[np.float32]] = {}
        texts_to_process = []
        indices_to_process = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if self.cache:
                cached_embedding = self.cache.get(text, self.model_name)
                if cached_embedding is not None:
                    result_embeddings[i] = cached_embedding
                    continue

            # Mark for processing
            texts_to_process.append(text)
            indices_to_process.append(i)

        # Process uncached texts
        if texts_to_process:
            try:
                new_embeddings = self.model_manager.model.encode(texts_to_process, batch_size=self.batch_size, normalize_embeddings=True)

                # Store in cache and update results
                for idx, embedding in zip(indices_to_process, new_embeddings):
                    if self.cache:
                        self.cache.set(texts[idx], self.model_name, embedding)
                    result_embeddings[idx] = embedding

            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                # Fill with zeros for failed embeddings
                zero_embedding = np.zeros(self.dimensions or 256, dtype=np.float32)
                for idx in indices_to_process:
                    result_embeddings[idx] = zero_embedding

        # Convert dictionary back to ordered list
        return [result_embeddings[i] for i in range(len(texts))]

    def generate_query_embedding(self, query: str) -> NDArray[np.float32]:
        """Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Query embedding array

        """
        # Add query prefix
        prefixed_query = f"{self.query_prefix}{query}"

        # Generate embedding
        embeddings = self.generate_embeddings([prefixed_query])

        if embeddings:
            return embeddings[0]
        else:
            # Return zero embedding as fallback
            return np.zeros(self.dimensions or 256, dtype=np.float32)

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self.cache and self.cache.cache_dir.exists():
            for cache_file in self.cache.cache_dir.glob("*.npy"):
                cache_file.unlink(missing_ok=True)
