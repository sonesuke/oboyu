"""Embedding generation service."""

import hashlib
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from oboyu.common.huggingface_utils import (
    get_fallback_models,
)
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
            batch_size: Batch size for embedding generation
            max_seq_length: Maximum sequence length
            query_prefix: Prefix for search queries
            use_onnx: Whether to use ONNX optimization
            use_cache: Whether to use embedding cache
            onnx_quantization_config: ONNX quantization configuration
            onnx_optimization_level: ONNX optimization level

        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.query_prefix = query_prefix
        self.use_onnx = use_onnx
        self.use_cache = use_cache

        # Initialize model manager (lazy loading, actual errors occur when model property is accessed)
        try:
            self.model_manager = EmbeddingModelManager(
                model_name=model_name,
                models_dir=EMBEDDING_MODELS_DIR,
                max_seq_length=max_seq_length,
                use_onnx=use_onnx,
                quantization_config=onnx_quantization_config,
                optimization_level=onnx_optimization_level,
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model manager for {model_name}: {e}")
            # Suggest fallback models
            fallbacks = get_fallback_models("embedding")
            if fallbacks:
                fallback_msg = "Suggested fallback embedding models:\n"
                for model_id, description in fallbacks[:3]:
                    fallback_msg += f"• {model_id}: {description}\n"
                logger.error(fallback_msg)
            raise RuntimeError(f"Failed to initialize embedding service with model '{model_name}': {e}") from e

        # Initialize cache if enabled
        self.cache = EmbeddingCache() if use_cache else None

    @property
    def dimensions(self) -> Optional[int]:
        """Get embedding dimensions."""
        return self.model_manager.get_dimensions()

    def generate_embeddings(self, texts: List[str], progress_callback: Optional[Callable[[str, int, int], None]] = None) -> List[NDArray[np.float32]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings for
            progress_callback: Optional callback for progress updates

        Returns:
            List of embedding arrays

        """
        if not texts:
            return []

        result_embeddings, texts_to_process, indices_to_process = self._prepare_embedding_batch(texts)

        if texts_to_process:
            try:
                new_embeddings = self._process_embeddings_batch(texts_to_process, progress_callback)
                self._store_embeddings_results(texts, indices_to_process, new_embeddings, result_embeddings)
            except Exception as e:
                self._handle_embedding_error(e, indices_to_process, result_embeddings)

        # Convert dictionary back to ordered list
        return [result_embeddings[i] for i in range(len(texts))]

    def _prepare_embedding_batch(self, texts: List[str]) -> tuple[Dict[int, NDArray[np.float32]], List[str], List[int]]:
        """Prepare texts for embedding generation by checking cache."""
        result_embeddings: Dict[int, NDArray[np.float32]] = {}
        texts_to_process = []
        indices_to_process = []

        for i, text in enumerate(texts):
            if self.cache:
                cached_embedding = self.cache.get(text, self.model_name)
                if cached_embedding is not None and hasattr(cached_embedding, "shape") and cached_embedding.shape != ():
                    result_embeddings[i] = cached_embedding
                    continue

            texts_to_process.append(text)
            indices_to_process.append(i)

        return result_embeddings, texts_to_process, indices_to_process

    def _process_embeddings_batch(self, texts_to_process: List[str], progress_callback: Optional[Callable[[str, int, int], None]]) -> NDArray[np.float32]:
        """Process a batch of texts to generate embeddings."""
        logger.info(f"Processing {len(texts_to_process)} uncached texts with batch_size={self.batch_size}")

        num_batches = (len(texts_to_process) + self.batch_size - 1) // self.batch_size

        if progress_callback:
            progress_callback("embedding", 0, num_batches)

        # Use optimized processing for large batches
        if progress_callback and len(texts_to_process) > self.batch_size * 2:
            return self._process_large_batch(texts_to_process, num_batches, progress_callback)
        else:
            new_embeddings = self.model_manager.model.encode(texts_to_process, batch_size=self.batch_size, normalize_embeddings=True)
            if progress_callback:
                progress_callback("embedding", num_batches, num_batches)
            return new_embeddings

    def _process_large_batch(self, texts_to_process: List[str], num_batches: int, progress_callback: Callable[[str, int, int], None]) -> NDArray[np.float32]:
        """Process large batches with optimized progress reporting."""
        all_embeddings = []
        progress_report_interval = max(1, num_batches // 10)

        for batch_idx in range(0, len(texts_to_process), self.batch_size):
            batch_texts = texts_to_process[batch_idx : batch_idx + self.batch_size]
            batch_embeddings = self.model_manager.model.encode(batch_texts, batch_size=self.batch_size, normalize_embeddings=True)
            all_embeddings.append(batch_embeddings)

            current_batch = (batch_idx // self.batch_size) + 1
            if current_batch % progress_report_interval == 0 or current_batch == num_batches:
                progress_callback("embedding", current_batch, num_batches)

        return np.concatenate(all_embeddings, axis=0)

    def _store_embeddings_results(
        self,
        texts: List[str],
        indices_to_process: List[int],
        new_embeddings: NDArray[np.float32],
        result_embeddings: Dict[int, NDArray[np.float32]],
    ) -> None:
        """Store embedding results in cache and result dictionary."""
        if len(indices_to_process) == 1 and new_embeddings.ndim == 1:
            idx = indices_to_process[0]
            embedding = new_embeddings
            if self.cache:
                self.cache.set(texts[idx], self.model_name, embedding)
            result_embeddings[idx] = embedding
        else:
            for i, idx in enumerate(indices_to_process):
                embedding = new_embeddings[i]
                if self.cache:
                    self.cache.set(texts[idx], self.model_name, embedding)
                result_embeddings[idx] = embedding

    def _handle_embedding_error(self, error: Exception, indices_to_process: List[int], result_embeddings: Dict[int, NDArray[np.float32]]) -> None:
        """Handle errors during embedding generation."""
        if isinstance(error, RuntimeError) and "Failed to load embedding model" in str(error):
            logger.error(f"Model loading failed: {error}")
            fallbacks = get_fallback_models("embedding")
            if fallbacks:
                fallback_msg = "\n\nSuggested fallback models:\n"
                for model_id, description in fallbacks[:3]:
                    fallback_msg += f"• {model_id}: {description}\n"
                logger.error(fallback_msg)

            raise RuntimeError(f"Embedding service failed to load model '{self.model_name}'. {str(error)}") from error
        else:
            logger.error(f"Failed to generate embeddings: {error}")
            zero_embedding = np.zeros(self.dimensions or 256, dtype=np.float32)
            for idx in indices_to_process:
                result_embeddings[idx] = zero_embedding

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
            result = embeddings[0]
            return result.astype(np.float32)
        else:
            # Return zero embedding as fallback
            return np.zeros(self.dimensions or 256, dtype=np.float32)

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self.cache and self.cache.cache_dir.exists():
            for cache_file in self.cache.cache_dir.glob("*.npy"):
                cache_file.unlink(missing_ok=True)
