"""Legacy import compatibility for onnx_converter module."""

# Re-export from the new location
from oboyu.common.onnx_converter import *  # noqa: F403, F405

__all__ = ["convert_to_onnx", "convert_cross_encoder_to_onnx"]  # noqa: F405
