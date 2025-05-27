"""Embedding generation for Oboyu indexer.

This module handles the generation of embeddings for document chunks
using the Ruri v3 model with specialized handling for Japanese content.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from torch import Tensor

from oboyu.common.paths import EMBEDDING_CACHE_DIR, EMBEDDING_MODELS_DIR
from oboyu.indexer.onnx_converter import ONNXEmbeddingModel, get_or_convert_onnx_model
from oboyu.indexer.processor import Chunk

# Global model cache to avoid reloading models in the same process
_MODEL_CACHE = {}

# Silence SentenceTransformer logging (INFO level is too verbose)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


class EmbeddingCache:
    """Cache for document embeddings to avoid regeneration."""

    def __init__(self, cache_dir: Union[str, Path] = EMBEDDING_CACHE_DIR) -> None:
        """Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings (defaults to XDG cache path)

        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def get(self, text: str, model_name: str) -> Optional[NDArray[np.float32]]:
        """Retrieve embedding from cache.

        Args:
            text: Text to get embedding for
            model_name: Name of the embedding model

        Returns:
            Cached embedding or None if not found

        """
        cache_key = self._get_cache_key(text, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    # Convert the list back to numpy array and cast to correct type
                    return cast(NDArray[np.float32], np.array(data, dtype=np.float32))
            except (json.JSONDecodeError, IOError):
                # If the cache file is corrupted, return None
                return None

        return None

    def set(self, text: str, model_name: str, embedding: NDArray[np.float32]) -> None:
        """Store embedding in cache.

        Args:
            text: Text to store embedding for
            model_name: Name of the embedding model
            embedding: The embedding vector

        """
        cache_key = self._get_cache_key(text, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        with open(cache_file, "w") as f:
            # Convert numpy array to list for JSON serialization
            json.dump(embedding.tolist(), f)

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate a deterministic cache key for a text and model.

        Args:
            text: Input text
            model_name: Model name

        Returns:
            Cache key as string

        """
        # Create a unique hash based on text and model name
        # Using hashlib.sha256 which is more secure than MD5
        text_bytes = text.encode("utf-8")
        model_bytes = model_name.encode("utf-8")
        combined = text_bytes + b"_" + model_bytes
        return hashlib.sha256(combined).hexdigest()


class EmbeddingGenerator:
    """Generator for document embeddings using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-30m",
        device: str = "cpu",
        batch_size: int = 128,
        max_seq_length: int = 8192,
        query_prefix: str = "検索クエリ: ",
        use_cache: bool = True,
        cache_dir: Union[str, Path] = EMBEDDING_CACHE_DIR,
        model_dir: Union[str, Path] = EMBEDDING_MODELS_DIR,
        use_onnx: bool = True,
        onnx_quantization_config: Optional[Dict[str, Any]] = None,
        onnx_optimization_level: str = "none",
    ) -> None:
        """Initialize the embedding generator.

        Args:
            model_name: Name of the pretrained model to use
            device: Device to use for embeddings (cpu/cuda)
            batch_size: Batch size for embedding generation
            max_seq_length: Maximum sequence length for the model
            query_prefix: Prefix to add to search queries
            use_cache: Whether to use embedding cache
            cache_dir: Directory to store cached embeddings (defaults to XDG cache path)
            model_dir: Directory to store downloaded models (defaults to XDG data path)
            use_onnx: Whether to use ONNX optimization for faster inference
            onnx_quantization_config: Optional ONNX quantization configuration
            onnx_optimization_level: ONNX graph optimization level ("none", "basic", "extended", "all")

        Note:
            The model is loaded lazily on first use to improve startup performance.

        """
        # Store configuration for lazy loading
        self.use_onnx = use_onnx and device == "cpu"  # ONNX is most beneficial for CPU
        self.device = device
        self.model_dir = model_dir
        self.onnx_quantization_config = onnx_quantization_config or {"enabled": True, "weight_type": "uint8"}
        self.onnx_optimization_level = onnx_optimization_level
        self._model: Optional[Any] = None  # Lazy-loaded model

        # Include quantization in cache key if enabled
        quant_suffix = ""
        if self.use_onnx and self.onnx_quantization_config.get("enabled", True):
            quant_suffix = f"_quant_{self.onnx_quantization_config.get('weight_type', 'uint8')}"
        self._cache_key = f"{model_name}_{device}_{max_seq_length}_{'onnx' if self.use_onnx else 'torch'}{quant_suffix}"

        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.max_seq_length = max_seq_length

        # Set up cache
        self.use_cache = use_cache
        if use_cache:
            self.cache = EmbeddingCache(cache_dir)

        # Dimensions for this model (will be set when model is loaded)
        self._dimensions: Optional[int] = None

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Get the model, loading it lazily if not already loaded."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions, loading the model if necessary."""
        if self._dimensions is None:
            # Trigger model loading to get dimensions
            _ = self.model
        return self._dimensions or 256  # Fallback for type safety

    def _load_model(self) -> None:
        """Load the model (ONNX or PyTorch) and cache it."""
        # Use global model cache to avoid reloading models in the same process
        if self._cache_key not in _MODEL_CACHE:
            model: Any  # Type hint to allow both model types
            if self.use_onnx:
                # Load or convert to ONNX model
                # ONNX models are cached separately in XDG cache directory
                apply_quantization = self.onnx_quantization_config.get("enabled", True)
                onnx_path = get_or_convert_onnx_model(self.model_name, apply_quantization=apply_quantization, quantization_config=self.onnx_quantization_config)
                model = ONNXEmbeddingModel(
                    onnx_path,
                    max_seq_length=self.max_seq_length,
                    optimization_level=self.onnx_optimization_level,
                )
            else:
                # Load the model (with model_dir as cache directory)
                model = SentenceTransformer(self.model_name, device=self.device, cache_folder=str(self.model_dir))
                model.max_seq_length = self.max_seq_length
            _MODEL_CACHE[self._cache_key] = model

        self._model = _MODEL_CACHE[self._cache_key]
        # Set dimensions after loading
        self._dimensions = self._model.get_sentence_embedding_dimension()

    def generate_embeddings(
        self, chunks: List[Chunk], progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Tuple[str, str, NDArray[np.float32], datetime]]:
        """Generate embeddings for a list of document chunks.

        Args:
            chunks: List of document chunks
            progress_callback: Optional callback for progress updates
                               (chunks_processed, total_chunks, status)

        Returns:
            List of (id, chunk_id, embedding, timestamp) tuples

        """
        # Prepare texts for embedding
        texts_to_embed = []
        chunk_indices = []
        total_chunk_count = len(chunks)
        processed_count = 0
        cache_hit_count = 0

        for i, chunk in enumerate(chunks):
            if chunk.prefix_content is None:
                # Skip chunks with no content
                processed_count += 1
                if progress_callback:
                    progress_callback(processed_count, total_chunk_count, f"Skipped empty chunk {processed_count}/{total_chunk_count}")
                continue

            # Check cache first if enabled
            embedding = None
            if self.use_cache:
                embedding = self.cache.get(chunk.prefix_content, self.model_name)
                if embedding is not None:
                    cache_hit_count += 1

            if embedding is None:
                # Need to generate embedding
                texts_to_embed.append(chunk.prefix_content)
                chunk_indices.append(i)
            else:
                # Count cached chunks as processed
                processed_count += 1
                if progress_callback:
                    progress_callback(
                        processed_count, total_chunk_count, f"Using cached embedding {processed_count}/{total_chunk_count} ({cache_hit_count} cache hits)"
                    )

        # Generate new embeddings if needed
        new_embeddings = []
        if texts_to_embed:
            
            # Process in batches
            for i in range(0, len(texts_to_embed), self.batch_size):
                batch_texts = texts_to_embed[i : i + self.batch_size]

                # Update progress before batch processing
                if progress_callback:
                    progress_callback(processed_count, total_chunk_count, f"Generating batch of {len(batch_texts)} embeddings...")

                # Process batch
                batch_embeddings = self.model.encode(batch_texts, normalize_embeddings=True)

                for j, embedding in enumerate(batch_embeddings):
                    chunk_idx = chunk_indices[i + j]
                    chunk = chunks[chunk_idx]

                    # Cache the embedding if enabled
                    if self.use_cache and chunk.prefix_content is not None:
                        self.cache.set(chunk.prefix_content, self.model_name, cast(NDArray[np.float32], embedding))

                    # Add to results
                    new_embeddings.append(
                        (
                            f"{uuid.uuid4()}",  # Generate unique ID for the embedding
                            chunk.id,  # Reference to the chunk
                            np.asarray(embedding, dtype=np.float32),  # The embedding vector - ensure numpy array
                            datetime.now(),  # Timestamp
                        )
                    )

                    # Update progress per embedding
                    processed_count += 1
                    if progress_callback:
                        progress_callback(processed_count, total_chunk_count, f"Generated embedding {processed_count}/{total_chunk_count}")

        # Collect pre-cached embeddings
        cached_embeddings = []
        if self.use_cache:
            for i, chunk in enumerate(chunks):
                if i not in chunk_indices and chunk.prefix_content is not None:
                    # This chunk was in the cache
                    cached_emb = self.cache.get(chunk.prefix_content, self.model_name)
                    if cached_emb is not None:
                        cached_embeddings.append(
                            (
                                f"{uuid.uuid4()}",  # Generate unique ID for the embedding
                                chunk.id,  # Reference to the chunk
                                cached_emb,  # The embedding vector - already properly typed from get()
                                datetime.now(),  # Timestamp
                            )
                        )

        # Final progress update
        if progress_callback:
            progress_callback(
                total_chunk_count, total_chunk_count, f"Completed embedding generation: {len(new_embeddings)} new, {len(cached_embeddings)} cached"
            )

        # Combine new and cached embeddings
        return new_embeddings + cached_embeddings

    def generate_query_embedding(self, query: str) -> NDArray[np.float32]:
        """Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector

        """
        # Add the query prefix
        prefixed_query = f"{self.query_prefix}{query}"

        # Check cache first if enabled
        if self.use_cache:
            cached_embedding = self.cache.get(prefixed_query, self.model_name)
            if cached_embedding is not None:
                return cached_embedding

        # Generate the embedding
        embedding = self.model.encode(prefixed_query, normalize_embeddings=True)

        # Convert tensor to numpy array if needed
        if isinstance(embedding, Tensor):
            embedding_array = cast(NDArray[np.float32], embedding.cpu().numpy())
        else:
            embedding_array = cast(NDArray[np.float32], embedding)

        # Cache it
        if self.use_cache:
            self.cache.set(prefixed_query, self.model_name, embedding_array)

        return embedding_array
