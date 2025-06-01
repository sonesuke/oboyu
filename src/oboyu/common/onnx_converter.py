"""ONNX conversion and optimization for SentenceTransformer models.

This module handles the conversion of SentenceTransformer models to ONNX format
for faster inference, especially beneficial for CPU-based deployments.

This is a compatibility module that re-exports functionality from the onnx package.
"""

# Re-export all functionality from the onnx package for backward compatibility
from .onnx import (
    QUANTIZATION_AVAILABLE,
    ONNXCrossEncoderModel,
    ONNXEmbeddingModel,
    QuantType,
    convert_cross_encoder_to_onnx,
    convert_to_onnx,
    get_or_convert_cross_encoder_onnx_model,
    get_or_convert_onnx_model,
    quantize_model_dynamic,
)

__all__ = [
    "QUANTIZATION_AVAILABLE",
    "QuantType",
    "quantize_model_dynamic",
    "ONNXEmbeddingModel",
    "convert_to_onnx",
    "get_or_convert_onnx_model",
    "ONNXCrossEncoderModel",
    "convert_cross_encoder_to_onnx",
    "get_or_convert_cross_encoder_onnx_model",
]
