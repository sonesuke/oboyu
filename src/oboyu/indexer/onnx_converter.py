"""ONNX conversion and optimization for SentenceTransformer models.

This module handles the conversion of SentenceTransformer models to ONNX format
for faster inference, especially beneficial for CPU-based deployments.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore[attr-defined]

from oboyu.common.paths import EMBEDDING_CACHE_DIR

logger = logging.getLogger(__name__)


class ONNXEmbeddingModel:
    """ONNX-optimized embedding model for faster inference."""

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        max_seq_length: int = 8192,
        pooling_strategy: str = "mean",
    ) -> None:
        """Initialize ONNX embedding model.

        Args:
            model_path: Path to ONNX model file
            tokenizer_path: Path to tokenizer directory (defaults to model_path parent)
            max_seq_length: Maximum sequence length
            pooling_strategy: Pooling strategy for embeddings ("mean", "max", "cls")

        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path.parent
        self.max_seq_length = max_seq_length
        self.pooling_strategy = pooling_strategy

        # Set up ONNX runtime options for optimization
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Adjust based on CPU cores

        # Load ONNX model
        self.session = InferenceSession(str(self.model_path), sess_options)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
        
        # Get model dimensions from output shape
        output_shape = self.session.get_outputs()[0].shape
        self.dimensions = output_shape[-1] if len(output_shape) > 1 else None

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 8,
        normalize_embeddings: bool = True,
    ) -> NDArray[np.float32]:
        """Encode sentences to embeddings using ONNX model.

        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize embeddings

        Returns:
            Embeddings as numpy array

        """
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
            
            # Run inference
            outputs = self.session.run(None, {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            })
            
            # Apply pooling
            embeddings = self._pool_embeddings(
                outputs[0],  # token embeddings
                inputs["attention_mask"],
            )
            
            if normalize_embeddings:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        result = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        
        # Return single embedding if input was single sentence
        if len(sentences) == 1:
            return np.asarray(result[0], dtype=np.float32)
        else:
            return np.asarray(result, dtype=np.float32)

    def _pool_embeddings(
        self,
        token_embeddings: NDArray[np.float32],
        attention_mask: NDArray[np.int64],
    ) -> NDArray[np.float32]:
        """Apply pooling strategy to token embeddings.

        Args:
            token_embeddings: Token-level embeddings [batch, seq_len, dim]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Pooled embeddings [batch, dim]

        """
        if self.pooling_strategy == "cls":
            # Use CLS token
            return np.asarray(token_embeddings[:, 0, :], dtype=np.float32)
        elif self.pooling_strategy == "max":
            # Max pooling with attention mask
            masked = token_embeddings * attention_mask[:, :, np.newaxis]
            return np.asarray(np.max(masked, axis=1), dtype=np.float32)
        else:  # mean pooling
            # Mean pooling with attention mask
            masked = token_embeddings * attention_mask[:, :, np.newaxis]
            sum_embeddings = np.sum(masked, axis=1)
            sum_mask = np.sum(attention_mask, axis=1, keepdims=True)
            return np.asarray(sum_embeddings / np.maximum(sum_mask, 1e-9), dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of sentence embeddings.

        Returns:
            Embedding dimension

        """
        return self.dimensions or 768  # Default to 768 if not detected


def convert_to_onnx(
    model_name: str,
    output_dir: Union[str, Path],
    opset_version: int = 14,
    optimize: bool = True,
) -> Path:
    """Convert a SentenceTransformer model to ONNX format.

    Args:
        model_name: Name or path of the SentenceTransformer model
        output_dir: Directory to save ONNX model
        opset_version: ONNX opset version
        optimize: Whether to optimize the ONNX model

    Returns:
        Path to the saved ONNX model

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting {model_name} to ONNX format...")
    
    # Load the model - force CPU to avoid MPS issues
    model = SentenceTransformer(model_name, device="cpu")
    
    # Get the transformer model and ensure it's on CPU
    transformer = model[0].auto_model
    transformer = transformer.to("cpu")
    
    # Prepare dummy input on CPU
    tokenizer = model.tokenizer
    dummy_input = tokenizer(
        "This is a test sentence for ONNX conversion.",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.max_seq_length,
    )
    # Ensure inputs are on CPU
    dummy_input = {k: v.to("cpu") for k, v in dummy_input.items()}
    
    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        transformer,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
    )
    
    # Optimize if requested
    if optimize:
        logger.info("Optimizing ONNX model...")
        try:
            from onnxruntime.transformers import optimizer
            
            optimized_path = output_dir / "model_optimized.onnx"
            optimizer.optimize_model(
                str(onnx_path),
                str(optimized_path),
                optimization_options=optimizer.FusionOptions("bert"),
            )
            # Check if optimization succeeded
            if optimized_path.exists() and optimized_path.stat().st_size > 0:
                onnx_path = optimized_path
            else:
                logger.warning("Optimization failed, using non-optimized model")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}, using non-optimized model")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save config
    config = {
        "max_seq_length": model.max_seq_length,
        "pooling_strategy": "mean",  # Default for most models
        "embedding_dimension": model.get_sentence_embedding_dimension(),
    }
    with open(output_dir / "onnx_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"ONNX model saved to {onnx_path}")
    return onnx_path


def get_or_convert_onnx_model(
    model_name: str,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Get ONNX model path, converting if necessary.

    Args:
        model_name: Name of the SentenceTransformer model
        cache_dir: Directory to cache ONNX models (defaults to XDG cache path)

    Returns:
        Path to ONNX model file

    """
    if cache_dir is None:
        cache_dir = EMBEDDING_CACHE_DIR / "models"
    else:
        cache_dir = Path(cache_dir)
    model_dir = cache_dir / "onnx" / model_name.replace("/", "_")
    
    # Try optimized model first
    onnx_path = model_dir / "model_optimized.onnx"
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        return onnx_path
    
    # Try non-optimized model
    onnx_path = model_dir / "model.onnx"
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        return onnx_path
    
    # Convert if not found
    logger.info(f"ONNX model not found, converting {model_name}...")
    onnx_path = convert_to_onnx(model_name, model_dir, optimize=False)  # Disable optimization to avoid issues
    
    return onnx_path


class ONNXCrossEncoderModel:
    """ONNX-optimized Cross-Encoder model for reranking."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        max_seq_length: int = 512,
    ) -> None:
        """Initialize ONNX Cross-Encoder model.
        
        Args:
            model_path: Path to ONNX model file
            tokenizer_path: Path to tokenizer directory (defaults to model_path parent)
            max_seq_length: Maximum sequence length
        
        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path.parent
        self.max_seq_length = max_seq_length
        
        # Set up ONNX runtime options for optimization
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Adjust based on CPU cores
        
        # Load ONNX model
        self.session = InferenceSession(str(self.model_path), sess_options)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
    
    def predict(
        self,
        queries: Union[str, List[str]],
        documents: Union[str, List[str]],
        batch_size: int = 8,
    ) -> NDArray[np.float32]:
        """Predict relevance scores for query-document pairs.
        
        Args:
            queries: Single query or list of queries
            documents: Single document or list of documents
            batch_size: Batch size for prediction
            
        Returns:
            Relevance scores as numpy array
        
        """
        # Handle single inputs
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(documents, str):
            documents = [documents]
        
        if len(queries) != len(documents):
            raise ValueError(f"Number of queries ({len(queries)}) must match number of documents ({len(documents)})")
        
        all_scores = []
        
        # Process in batches
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            
            # Tokenize pairs
            inputs = self.tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
            
            # Run inference
            outputs = self.session.run(None, {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            })
            
            # Extract logits
            logits = outputs[0]
            
            # Apply sigmoid to get probabilities
            scores = 1 / (1 + np.exp(-logits[:, 0]))  # Take positive class logit
            all_scores.extend(scores)
        
        return np.array(all_scores, dtype=np.float32)


def convert_cross_encoder_to_onnx(
    model_name: str,
    output_dir: Union[str, Path],
    opset_version: int = 14,
    optimize: bool = True,
) -> Path:
    """Convert a Cross-Encoder model to ONNX format.
    
    Args:
        model_name: Name or path of the Cross-Encoder model
        output_dir: Directory to save ONNX model
        opset_version: ONNX opset version
        optimize: Whether to optimize the ONNX model
        
    Returns:
        Path to the saved ONNX model
    
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting Cross-Encoder {model_name} to ONNX format...")
    
    # Load the model and tokenizer
    if "cross-encoder" in model_name.lower() or "reranker" in model_name.lower():
        # Use CrossEncoder class if available
        try:
            ce_model = CrossEncoder(model_name, device="cpu", trust_remote_code=True)
            model = ce_model.model
            tokenizer = ce_model.tokenizer
            max_length = ce_model.max_length
        except Exception:
            # Fallback to direct loading
            model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            max_length = 512
    else:
        # Direct loading for other models
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        max_length = 512
    
    # Ensure model is on CPU
    model = model.to("cpu")
    model.eval()
    
    # Prepare dummy input
    dummy_input = tokenizer(
        "This is a query",
        "This is a document to rank",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    # Ensure inputs are on CPU
    dummy_input = {k: v.to("cpu") for k, v in dummy_input.items()}
    
    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
    )
    
    # Optimize if requested
    if optimize:
        logger.info("Optimizing ONNX model...")
        try:
            from onnxruntime.transformers import optimizer
            
            optimized_path = output_dir / "model_optimized.onnx"
            optimizer.optimize_model(
                str(onnx_path),
                str(optimized_path),
                optimization_options=optimizer.FusionOptions("bert"),
            )
            # Check if optimization succeeded
            if optimized_path.exists() and optimized_path.stat().st_size > 0:
                onnx_path = optimized_path
            else:
                logger.warning("Optimization failed, using non-optimized model")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}, using non-optimized model")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save config
    config = {
        "max_seq_length": max_length,
        "model_type": "cross-encoder",
    }
    with open(output_dir / "onnx_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"ONNX Cross-Encoder model saved to {onnx_path}")
    return onnx_path


def get_or_convert_cross_encoder_onnx_model(
    model_name: str,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Get ONNX Cross-Encoder model path, converting if necessary.
    
    Args:
        model_name: Name of the Cross-Encoder model
        cache_dir: Directory to cache ONNX models (defaults to XDG cache path)
        
    Returns:
        Path to ONNX model file
    
    """
    if cache_dir is None:
        cache_dir = EMBEDDING_CACHE_DIR / "models"
    else:
        cache_dir = Path(cache_dir)
    model_dir = cache_dir / "onnx" / model_name.replace("/", "_")
    
    # Try optimized model first
    onnx_path = model_dir / "model_optimized.onnx"
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        return onnx_path
    
    # Try non-optimized model
    onnx_path = model_dir / "model.onnx"
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        return onnx_path
    
    # Convert if not found
    logger.info(f"ONNX Cross-Encoder model not found, converting {model_name}...")
    onnx_path = convert_cross_encoder_to_onnx(model_name, model_dir, optimize=False)  # Disable optimization to avoid issues
    
    return onnx_path
