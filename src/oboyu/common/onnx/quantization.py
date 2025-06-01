"""ONNX model quantization utilities."""

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Try to import quantization tools
try:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    QuantType = None
    quantize_dynamic = None
    logger.warning("ONNX Runtime quantization tools not available. Quantization will be disabled.")


__all__ = ["QUANTIZATION_AVAILABLE", "QuantType", "quantize_model_dynamic"]


def quantize_model_dynamic(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    weight_type: str = "uint8",
) -> Path:
    """Apply dynamic quantization to an ONNX model.

    Dynamic quantization quantizes the weights to int8 while keeping activations in fp32.
    This provides good performance improvement with minimal accuracy loss.

    Args:
        model_path: Path to the ONNX model file
        output_path: Path for the quantized model (defaults to model_path with .quantized suffix)
        weight_type: Weight quantization type ("uint8" or "int8")

    Returns:
        Path to the quantized model

    Raises:
        RuntimeError: If quantization tools are not available

    """
    if not QUANTIZATION_AVAILABLE:
        raise RuntimeError("ONNX Runtime quantization tools are not available. Please install onnxruntime-tools.")

    model_path = Path(model_path)
    if output_path is None:
        # Default to model_quantized.onnx in the same directory
        output_path = model_path.parent / "model_quantized.onnx"
    else:
        output_path = Path(output_path)

    # Determine quantization type
    quant_type = QuantType.QUInt8 if weight_type == "uint8" else QuantType.QInt8

    logger.info(f"Quantizing model {model_path} to {output_path} with {weight_type} weights...")

    try:
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(output_path),
            weight_type=quant_type,
        )

        # Verify the quantized model exists and has reasonable size
        if output_path.exists() and output_path.stat().st_size > 0:
            original_size = model_path.stat().st_size
            quantized_size = output_path.stat().st_size
            reduction = (1 - quantized_size / original_size) * 100
            logger.info(f"Quantization complete. Size reduced by {reduction:.1f}% ({original_size} -> {quantized_size} bytes)")
            return output_path
        else:
            raise RuntimeError("Quantization failed: output file is missing or empty")

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        # Clean up partial output if it exists
        if output_path.exists():
            output_path.unlink()
        raise
