"""Embedding generation for Oboyu indexer.

This module handles the generation of embeddings for document chunks
using the Ruri v3 model with specialized handling for Japanese content.
"""

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from torch import Tensor

from oboyu.indexer.processor import Chunk


class EmbeddingCache:
    """Cache for document embeddings to avoid regeneration."""

    def __init__(self, cache_dir: Union[str, Path] = ".embedding_cache") -> None:
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
        batch_size: int = 8,
        max_seq_length: int = 8192,
        query_prefix: str = "検索クエリ: ",
        use_cache: bool = True,
        cache_dir: Union[str, Path] = ".embedding_cache",
    ) -> None:
        """Initialize the embedding generator.

        Args:
            model_name: Name of the pretrained model to use
            device: Device to use for embeddings (cpu/cuda)
            batch_size: Batch size for embedding generation
            max_seq_length: Maximum sequence length for the model
            query_prefix: Prefix to add to search queries
            use_cache: Whether to use embedding cache
            cache_dir: Directory to store cached embeddings

        """
        # Load the model
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length

        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix

        # Set up cache
        self.use_cache = use_cache
        if use_cache:
            self.cache = EmbeddingCache(cache_dir)

        # Dimensions for this model (Ruri v3-30m produces 256-dim embeddings)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, chunks: List[Chunk]) -> List[Tuple[str, str, NDArray[np.float32], datetime]]:
        """Generate embeddings for a list of document chunks.

        Args:
            chunks: List of document chunks

        Returns:
            List of (id, chunk_id, embedding, timestamp) tuples

        """
        # Prepare texts for embedding
        texts_to_embed = []
        chunk_indices = []

        for i, chunk in enumerate(chunks):
            if chunk.prefix_content is None:
                continue

            # Check cache first if enabled
            embedding = None
            if self.use_cache:
                embedding = self.cache.get(chunk.prefix_content, self.model_name)

            if embedding is None:
                # Need to generate embedding
                texts_to_embed.append(chunk.prefix_content)
                chunk_indices.append(i)

        # Generate new embeddings if needed
        new_embeddings = []
        if texts_to_embed:
            # Process in batches
            for i in range(0, len(texts_to_embed), self.batch_size):
                batch_texts = texts_to_embed[i:i + self.batch_size]
                batch_embeddings = self.model.encode(batch_texts, normalize_embeddings=True)

                for j, embedding in enumerate(batch_embeddings):
                    chunk_idx = chunk_indices[i + j]
                    chunk = chunks[chunk_idx]

                    # Cache the embedding if enabled
                    if self.use_cache and chunk.prefix_content is not None:
                        self.cache.set(chunk.prefix_content, self.model_name, cast(NDArray[np.float32], embedding))

                    # Add to results
                    new_embeddings.append((
                        f"{uuid.uuid4()}",  # Generate unique ID for the embedding
                        chunk.id,  # Reference to the chunk
                        cast(NDArray[np.float32], embedding),  # The embedding vector - cast to correct type
                        datetime.now(),  # Timestamp
                    ))

        # Collect pre-cached embeddings
        cached_embeddings = []
        if self.use_cache:
            for i, chunk in enumerate(chunks):
                if i not in chunk_indices and chunk.prefix_content is not None:
                    # This chunk was in the cache
                    cached_emb = self.cache.get(chunk.prefix_content, self.model_name)
                    if cached_emb is not None:
                        cached_embeddings.append((
                            f"{uuid.uuid4()}",  # Generate unique ID for the embedding
                            chunk.id,  # Reference to the chunk
                            cached_emb,  # The embedding vector - already properly typed from get()
                            datetime.now(),  # Timestamp
                        ))

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
