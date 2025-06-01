"""Legacy import compatibility for reranker module."""

# Re-export from the new location
from oboyu.indexer.models.reranker_service import (
    BaseReranker,
    CrossEncoderReranker,
    ONNXCrossEncoderReranker,
    RerankedResult,
    RerankerService,
    create_reranker,
)

__all__ = [
    "BaseReranker",
    "CrossEncoderReranker",
    "ONNXCrossEncoderReranker",
    "RerankedResult",
    "RerankerService",
    "create_reranker",
]
