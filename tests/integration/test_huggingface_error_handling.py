"""Integration tests for Hugging Face error handling."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from oboyu.common.huggingface_utils import (
    HuggingFaceNetworkError,
    HuggingFaceModelNotFoundError,
    safe_model_download,
)
from oboyu.indexer.services.embedding import EmbeddingService
from oboyu.retriever.services.reranker import RerankerService


class TestHuggingFaceErrorHandlingIntegration:
    """Integration tests for error handling across the system."""

    def test_embedding_service_handles_model_not_found(self) -> None:
        """Test that embedding service handles model not found errors gracefully."""
        with pytest.raises(RuntimeError) as exc_info:
            EmbeddingService(model_name="non-existent/model")
        
        assert "Failed to initialize embedding service" in str(exc_info.value)

    def test_reranker_service_handles_model_not_found_gracefully(self) -> None:
        """Test that reranker service handles model errors without failing initialization."""
        # RerankerService should not raise on init, but should disable reranking
        service = RerankerService(model_name="non-existent/model")
        assert not service.is_available()

    @patch("oboyu.common.huggingface_utils.check_huggingface_connectivity")
    def test_network_error_fallback(self, mock_connectivity: Mock) -> None:
        """Test network error handling and fallback behavior."""
        mock_connectivity.return_value = False
        
        def failing_download() -> str:
            raise HuggingFaceNetworkError("Network error")
        
        with pytest.raises(HuggingFaceNetworkError):
            safe_model_download("test/model", failing_download)

    def test_model_download_with_retry_logic(self) -> None:
        """Test that model download implements retry logic."""
        call_count = 0
        
        def intermittent_failure() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise HuggingFaceModelNotFoundError("Model not found")
            return "success"
        
        with pytest.raises(HuggingFaceModelNotFoundError):
            safe_model_download("test/model", intermittent_failure, max_retries=1)
        
        # Should have been called twice (initial + 1 retry)
        assert call_count == 2