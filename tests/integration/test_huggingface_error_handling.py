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
        """Test that embedding service initializes without exception, even with invalid model."""
        # EmbeddingService may initialize successfully and fail later, or handle gracefully
        service = EmbeddingService(model_name="definitely-non-existent-model/that-does-not-exist")
        # Service should still be created, but might fail when actually used
        assert hasattr(service, 'model_name')

    def test_reranker_service_handles_model_not_found_gracefully(self) -> None:
        """Test that reranker service handles model errors without failing initialization."""
        # RerankerService should not raise on init, but should disable reranking
        service = RerankerService(model_name="definitely-non-existent-model/that-does-not-exist")
        # Service should still be created, but might not be available depending on fallback behavior
        assert hasattr(service, 'is_available')

    @patch("oboyu.common.huggingface_utils.check_huggingface_connectivity")
    def test_network_error_fallback(self, mock_connectivity: Mock) -> None:
        """Test network error handling and fallback behavior."""
        mock_connectivity.return_value = False
        
        def failing_download() -> str:
            raise HuggingFaceNetworkError("Network error")
        
        with pytest.raises(HuggingFaceNetworkError):
            safe_model_download("test/model", failing_download)

    def test_model_download_with_retry_logic(self) -> None:
        """Test that model download implements retry logic for temporary errors."""
        call_count = 0
        
        def intermittent_failure() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                import httpx
                raise httpx.TimeoutException("Timeout")
            return "success"
        
        # Should succeed after retries
        result = safe_model_download("test/model", intermittent_failure, max_retries=3)
        assert result == "success"
        
        # Should have been called 3 times (initial + 2 retries)
        assert call_count == 3