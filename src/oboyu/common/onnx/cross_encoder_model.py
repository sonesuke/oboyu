"""ONNX cross-encoder model implementation and conversion utilities."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .quantization import QUANTIZATION_AVAILABLE, quantize_model_dynamic

logger = logging.getLogger(__name__)


class ONNXCrossEncoderModel:
    """ONNX-optimized Cross-Encoder model for reranking."""

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        max_seq_length: int = 512,
        optimization_level: str = "none",
    ) -> None:
        """Initialize ONNX Cross-Encoder model.

        Args:
            model_path: Path to ONNX model file
            tokenizer_path: Path to tokenizer directory (defaults to model_path parent)
            max_seq_length: Maximum sequence length
            optimization_level: Graph optimization level ("none", "basic", "extended", "all")

        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path.parent
        self.max_seq_length = max_seq_length

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
            batch_queries = queries[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]

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
            outputs = self.session.run(
                None,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
            )

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
    apply_quantization: bool = True,
    quantization_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Convert a Cross-Encoder model to ONNX format.

    Args:
        model_name: Name or path of the Cross-Encoder model
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

    logger.debug(f"Converting Cross-Encoder {model_name} to ONNX format...")

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
            model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)  # type: ignore[no-untyped-call]
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            max_length = 512
    else:
        # Direct loading for other models
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)  # type: ignore[no-untyped-call]
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
        "max_seq_length": max_length,
        "model_type": "cross-encoder",
        "quantized": apply_quantization and QUANTIZATION_AVAILABLE and onnx_path.name == "model_quantized.onnx",
    }
    with open(output_dir / "onnx_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"ONNX Cross-Encoder model saved to {onnx_path}")
    return onnx_path


def get_or_convert_cross_encoder_onnx_model(
    model_name: str,
    cache_dir: Optional[Union[str, Path]] = None,
    apply_quantization: bool = True,
    quantization_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Get ONNX Cross-Encoder model path, converting if necessary.

    Args:
        model_name: Name of the Cross-Encoder model
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
        "reranker",
        cache_dir=Path(cache_dir) if cache_dir else None,
        apply_quantization=apply_quantization,
        quantization_config=quantization_config,
    )
