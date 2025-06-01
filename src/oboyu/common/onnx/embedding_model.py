"""ONNX embedding model implementation and conversion utilities."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .quantization import QUANTIZATION_AVAILABLE, quantize_model_dynamic

logger = logging.getLogger(__name__)


class ONNXEmbeddingModel:
    """ONNX-optimized embedding model for faster inference."""

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        max_seq_length: int = 8192,
        pooling_strategy: str = "mean",
        optimization_level: str = "none",
    ) -> None:
        """Initialize ONNX embedding model.

        Args:
            model_path: Path to ONNX model file
            tokenizer_path: Path to tokenizer directory (defaults to model_path parent)
            max_seq_length: Maximum sequence length
            pooling_strategy: Pooling strategy for embeddings ("mean", "max", "cls")
            optimization_level: Graph optimization level ("none", "basic", "extended", "all")

        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path.parent
        self.max_seq_length = max_seq_length
        self.pooling_strategy = pooling_strategy

        # Set up ONNX runtime options for optimization
        sess_options = SessionOptions()
        # Map string optimization level to GraphOptimizationLevel enum
        opt_level_map = {
            "none": GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        sess_options.graph_optimization_level = opt_level_map.get(optimization_level, GraphOptimizationLevel.ORT_DISABLE_ALL)
        # Use all available CPU cores for better performance
        num_threads = os.cpu_count() or 4
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1  # Single thread for operation scheduling
        logger.debug(f"ONNX session configured with {num_threads} intra-op threads, optimization level: {optimization_level}")

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
            batch = sentences[i : i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )

            # Run inference
            outputs = self.session.run(
                None,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
            )

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
    apply_quantization: bool = True,
    quantization_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Convert a SentenceTransformer model to ONNX format.

    Args:
        model_name: Name or path of the SentenceTransformer model
        output_dir: Directory to save ONNX model
        opset_version: ONNX opset version
        optimize: Whether to optimize the ONNX model
        apply_quantization: Whether to apply dynamic quantization (default: True)
        quantization_config: Optional quantization configuration

    Returns:
        Path to the saved ONNX model

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Converting {model_name} to ONNX format...")

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

    # Apply quantization if requested and available
    if apply_quantization and QUANTIZATION_AVAILABLE:
        try:
            # Get quantization config
            quant_config = quantization_config or {}
            weight_type = quant_config.get("weight_type", "uint8")

            # Quantize the model
            quantized_path = quantize_model_dynamic(onnx_path, weight_type=weight_type)

            # Update path to quantized model
            onnx_path = quantized_path
        except Exception as e:
            logger.warning(f"Quantization failed, using non-quantized model: {e}")
            # Continue with non-quantized model
    elif apply_quantization and not QUANTIZATION_AVAILABLE:
        logger.warning("Quantization requested but tools not available. Using non-quantized model.")

    # Save config
    config = {
        "max_seq_length": model.max_seq_length,
        "pooling_strategy": "mean",  # Default for most models
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "quantized": apply_quantization and QUANTIZATION_AVAILABLE and onnx_path.name == "model_quantized.onnx",
    }
    with open(output_dir / "onnx_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"ONNX model saved to {onnx_path}")
    return onnx_path


def get_or_convert_onnx_model(
    model_name: str,
    cache_dir: Optional[Union[str, Path]] = None,
    apply_quantization: bool = True,
    quantization_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Get ONNX model path, converting if necessary.

    Args:
        model_name: Name of the SentenceTransformer model
        cache_dir: Directory to cache ONNX models (defaults to XDG cache path)
        apply_quantization: Whether to apply dynamic quantization (default: True)
        quantization_config: Optional quantization configuration

    Returns:
        Path to ONNX model file

    """
    # Use the unified ONNXModelCache from model_manager
    from oboyu.common.model_manager import ONNXModelCache

    return ONNXModelCache.get_or_convert_onnx_model(
        model_name,
        "embedding",
        cache_dir=Path(cache_dir) if cache_dir else None,
        apply_quantization=apply_quantization,
        quantization_config=quantization_config,
    )
