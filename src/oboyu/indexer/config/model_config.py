"""Model-related configuration."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Model-related configuration."""

    # Embedding model settings
    embedding_model: str = "cl-nagoya/ruri-v3-30m"
    batch_size: int = 64
    max_seq_length: int = 8192
    use_onnx: bool = True

    # ONNX optimization settings
    onnx_quantization: Optional[Dict[str, object]] = None
    onnx_optimization_level: str = "extended"

    # Prefix scheme settings (Ruri v3's 1+3 prefix scheme)
    document_prefix: str = "検索文書: "
    query_prefix: str = "検索クエリ: "
    topic_prefix: str = "トピック: "
    general_prefix: str = ""

    # Reranker settings
    reranker_model: str = "cl-nagoya/ruri-reranker-small"
    use_reranker: bool = False
    reranker_use_onnx: bool = False
    reranker_device: str = "cpu"
    reranker_batch_size: int = 16
    reranker_max_length: int = 512

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        if self.onnx_quantization is None:
            self.onnx_quantization = {
                "enabled": True,
                "method": "dynamic",
                "weight_type": "uint8",
            }
