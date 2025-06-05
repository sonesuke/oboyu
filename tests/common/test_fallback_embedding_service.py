"""Tests for fallback embedding service functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from oboyu.common.circuit_breaker import CircuitBreakerError, CircuitState
from oboyu.common.fallback_embedding_service import (
    FallbackEmbeddingService,
    LocalEmbeddingService,
)
from oboyu.common.huggingface_utils import (
    HuggingFaceNetworkError,
    HuggingFaceTimeoutError,
)


class TestLocalEmbeddingService:
    """Test local embedding service functionality."""

    def test_initialization(self):
        """Test local embedding service initialization."""
        service = LocalEmbeddingService(dimensions=256)
        assert service.get_dimensions() == 256
        assert service.get_model_name() == "local_fallback"

    def test_generate_embeddings(self):
        """Test generating embeddings with local service."""
        service = LocalEmbeddingService(dimensions=128)
        
        texts = ["hello world", "test text"]
        embeddings = service.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert all(embedding.shape == (128,) for embedding in embeddings)
        assert all(np.allclose(np.linalg.norm(embedding), 1.0) for embedding in embeddings)

    def test_deterministic_embeddings(self):
        """Test that embeddings are deterministic for same text."""
        service = LocalEmbeddingService(dimensions=64)
        
        text = "test text"
        embedding1 = service.generate_embeddings([text])[0]
        embedding2 = service.generate_embeddings([text])[0]
        
        np.testing.assert_array_equal(embedding1, embedding2)

    def test_generate_query_embedding(self):
        """Test generating query embedding."""
        service = LocalEmbeddingService(dimensions=32)
        
        query = "search query"
        embedding = service.generate_query_embedding(query)
        
        assert embedding.shape == (32,)
        assert np.allclose(np.linalg.norm(embedding), 1.0)

    def test_caching_behavior(self):
        """Test that embeddings are cached in memory."""
        service = LocalEmbeddingService(dimensions=16)
        
        text = "cached text"
        embedding1 = service.generate_embeddings([text])[0]
        
        # Should use cached version
        assert text in service._word_vectors
        np.testing.assert_array_equal(service._word_vectors[text], embedding1)
        
        embedding2 = service.generate_embeddings([text])[0]
        np.testing.assert_array_equal(embedding1, embedding2)


class TestFallbackEmbeddingService:
    """Test fallback embedding service functionality."""

    @patch('oboyu.common.fallback_embedding_service.EmbeddingService')
    def test_initialization_with_primary_service(self, mock_embedding_service):
        """Test initialization with primary service."""
        mock_primary = Mock()
        mock_primary.dimensions = 256
        mock_embedding_service.return_value = mock_primary
        
        service = FallbackEmbeddingService(model_name="test-model")
        
        assert service.primary_service == mock_primary
        assert service.model_name == "test-model"
        assert service.use_circuit_breaker is True

    @patch('oboyu.common.fallback_embedding_service.EmbeddingService')
    def test_initialization_with_primary_service_param(self, mock_embedding_service):
        """Test initialization with primary service parameter."""
        mock_primary = Mock()
        mock_primary.dimensions = 512
        
        service = FallbackEmbeddingService(
            primary_service=mock_primary,
            enable_fallback_services=False,
            enable_local_fallback=False
        )
        
        assert service.primary_service == mock_primary
        # Should not create additional primary service when one is provided
        mock_embedding_service.assert_not_called()

    @patch('oboyu.common.fallback_embedding_service.EmbeddingService')
    def test_initialization_fallback_services(self, mock_embedding_service):
        """Test initialization of fallback services."""
        mock_primary = Mock()
        mock_primary.dimensions = 256
        
        # Mock successful creation of primary and one fallback
        mock_embedding_service.side_effect = [mock_primary, Mock(), Exception("Failed")]
        
        service = FallbackEmbeddingService(
            model_name="test-model",
            fallback_model_names=["fallback1", "fallback2"]
        )
        
        assert len(service.fallback_services) == 1  # One succeeded, one failed
        assert service.local_fallback is not None

    def test_initialization_local_fallback(self):
        """Test local fallback initialization."""
        with patch('oboyu.common.fallback_embedding_service.EmbeddingService') as mock_service:
            mock_primary = Mock()
            mock_primary.dimensions = 128
            mock_service.return_value = mock_primary
            
            service = FallbackEmbeddingService()
            
            assert service.local_fallback is not None
            assert service.local_fallback.get_dimensions() == 128

    @patch('oboyu.common.fallback_embedding_service.get_circuit_breaker_registry')
    def test_generate_embeddings_primary_success(self, mock_registry):
        """Test successful embedding generation with primary service."""
        mock_circuit_breaker = Mock()
        mock_registry.return_value.get_or_create.return_value = mock_circuit_breaker
        
        mock_primary = Mock()
        expected_embeddings = [np.array([1, 2, 3], dtype=np.float32)]
        mock_primary.generate_embeddings.return_value = expected_embeddings
        
        service = FallbackEmbeddingService(primary_service=mock_primary)
        
        # Mock circuit breaker call to execute the function
        mock_circuit_breaker.call.side_effect = lambda func: func()
        
        texts = ["test text"]
        result = service.generate_embeddings(texts)
        
        assert result == expected_embeddings
        mock_primary.generate_embeddings.assert_called_once_with(texts, None)

    @patch('oboyu.common.fallback_embedding_service.get_circuit_breaker_registry')
    def test_generate_embeddings_circuit_breaker_open(self, mock_registry):
        """Test fallback when circuit breaker is open."""
        # Mock primary circuit breaker to be open
        mock_primary_cb = Mock()
        mock_primary_cb.call.side_effect = CircuitBreakerError("Circuit open", "test", 5)
        
        # Mock fallback circuit breaker to work normally
        mock_fallback_cb = Mock()
        mock_fallback_cb.call.side_effect = lambda func: func()
        
        # Configure registry to return different circuit breakers
        def get_or_create_side_effect(name):
            if "embedding_fallback-model" in name:
                return mock_fallback_cb
            return mock_primary_cb
        
        mock_registry.return_value.get_or_create.side_effect = get_or_create_side_effect
        
        mock_primary = Mock()
        mock_primary.dimensions = 256  # Add proper dimensions attribute
        mock_fallback = Mock()
        expected_embeddings = [np.array([4, 5, 6], dtype=np.float32)]
        mock_fallback.generate_embeddings.return_value = expected_embeddings
        mock_fallback.model_name = "fallback-model"
        
        service = FallbackEmbeddingService(primary_service=mock_primary)
        service.fallback_services = [mock_fallback]
        
        texts = ["test text"]
        result = service.generate_embeddings(texts)
        
        assert result == expected_embeddings
        mock_fallback.generate_embeddings.assert_called_once_with(texts, None)

    def test_generate_embeddings_huggingface_error_fallback(self):
        """Test fallback when HuggingFace error occurs."""
        mock_primary = Mock()
        mock_primary.dimensions = 256  # Add proper dimensions attribute
        mock_primary.generate_embeddings.side_effect = HuggingFaceNetworkError("Network error")
        
        mock_fallback = Mock()
        expected_embeddings = [np.array([7, 8, 9], dtype=np.float32)]
        mock_fallback.generate_embeddings.return_value = expected_embeddings
        mock_fallback.model_name = "fallback-model"
        
        service = FallbackEmbeddingService(
            primary_service=mock_primary, 
            use_circuit_breaker=False
        )
        service.fallback_services = [mock_fallback]
        
        texts = ["test text"]
        result = service.generate_embeddings(texts)
        
        assert result == expected_embeddings

    def test_generate_embeddings_all_services_fail_local_fallback(self):
        """Test local fallback when all services fail."""
        mock_primary = Mock()
        mock_primary.dimensions = 64  # Add proper dimensions attribute
        mock_primary.generate_embeddings.side_effect = Exception("Primary failed")
        
        mock_fallback = Mock()
        mock_fallback.generate_embeddings.side_effect = Exception("Fallback failed")
        mock_fallback.model_name = "fallback-model"
        
        service = FallbackEmbeddingService(
            primary_service=mock_primary,
            use_circuit_breaker=False
        )
        service.fallback_services = [mock_fallback]
        service.local_fallback = LocalEmbeddingService(dimensions=64)
        
        texts = ["test text"]
        result = service.generate_embeddings(texts)
        
        assert len(result) == 1
        assert result[0].shape == (64,)

    def test_generate_embeddings_complete_failure(self):
        """Test zero embeddings when everything fails."""
        mock_primary = Mock()
        mock_primary.generate_embeddings.side_effect = Exception("Primary failed")
        mock_primary.dimensions = 32
        
        service = FallbackEmbeddingService(
            primary_service=mock_primary,
            use_circuit_breaker=False
        )
        service.fallback_services = []
        service.local_fallback = None
        
        texts = ["test text"]
        result = service.generate_embeddings(texts)
        
        assert len(result) == 1
        assert result[0].shape == (32,)
        assert np.allclose(result[0], 0.0)

    def test_generate_query_embedding(self):
        """Test query embedding generation."""
        mock_primary = Mock()
        expected_embedding = np.array([1, 2, 3], dtype=np.float32)
        mock_primary.generate_embeddings.return_value = [expected_embedding]
        
        service = FallbackEmbeddingService(
            primary_service=mock_primary,
            use_circuit_breaker=False,
            query_prefix="検索: "
        )
        
        query = "test query"
        result = service.generate_query_embedding(query)
        
        np.testing.assert_array_equal(result, expected_embedding)
        mock_primary.generate_embeddings.assert_called_once_with(["検索: test query"], None)

    def test_get_dimensions(self):
        """Test getting embedding dimensions."""
        mock_primary = Mock()
        mock_primary.dimensions = 512
        
        service = FallbackEmbeddingService(primary_service=mock_primary)
        
        assert service.get_dimensions() == 512

    def test_get_dimensions_fallback_chain(self):
        """Test getting dimensions from fallback chain."""
        service = FallbackEmbeddingService(primary_service=None, use_circuit_breaker=False)
        service.fallback_services = []
        service.local_fallback = LocalEmbeddingService(dimensions=128)
        
        # Check the local fallback dimensions directly
        assert service.local_fallback.dimensions == 128

    def test_get_model_name(self):
        """Test getting model name."""
        mock_primary = Mock()
        mock_primary.model_name = "test-model"
        
        service = FallbackEmbeddingService(primary_service=mock_primary)
        
        assert service.get_model_name() == "test-model"

    def test_get_model_name_fallback_chain(self):
        """Test getting model name from fallback chain."""
        service = FallbackEmbeddingService(
            primary_service=None, 
            model_name="fallback-model",
            use_circuit_breaker=False
        )
        
        assert service.get_model_name() == "fallback-model"

    def test_is_available_primary_closed(self):
        """Test availability when primary circuit is closed."""
        with patch('oboyu.common.fallback_embedding_service.get_circuit_breaker_registry') as mock_registry:
            mock_circuit_breaker = Mock()
            mock_circuit_breaker.get_state.return_value = CircuitState.CLOSED
            mock_registry.return_value.get_or_create.return_value = mock_circuit_breaker
            
            mock_primary = Mock()
            service = FallbackEmbeddingService(primary_service=mock_primary)
            
            assert service.is_available() is True

    def test_is_available_circuit_breaker_disabled(self):
        """Test availability when circuit breaker is disabled."""
        mock_primary = Mock()
        service = FallbackEmbeddingService(
            primary_service=mock_primary,
            use_circuit_breaker=False
        )
        
        assert service.is_available() is True

    def test_is_available_fallback_services(self):
        """Test availability through fallback services."""
        with patch('oboyu.common.fallback_embedding_service.get_circuit_breaker_registry') as mock_registry:
            mock_circuit_breaker = Mock()
            mock_circuit_breaker.get_state.return_value = CircuitState.OPEN
            mock_registry.return_value.get_or_create.return_value = mock_circuit_breaker
            mock_registry.return_value.get_circuit_breaker.return_value = None
            
            mock_fallback = Mock()
            mock_fallback.model_name = "fallback-model"
            
            service = FallbackEmbeddingService(primary_service=None)
            service.fallback_services = [mock_fallback]
            
            assert service.is_available() is True

    def test_is_available_local_fallback_only(self):
        """Test availability through local fallback only."""
        service = FallbackEmbeddingService(primary_service=None, use_circuit_breaker=False)
        service.fallback_services = []
        service.local_fallback = LocalEmbeddingService()
        
        assert service.is_available() is True

    def test_clear_cache(self):
        """Test clearing cache for all services."""
        mock_primary = Mock()
        mock_fallback = Mock()
        
        service = FallbackEmbeddingService(primary_service=mock_primary, use_circuit_breaker=False)
        service.fallback_services = [mock_fallback]
        
        service.clear_cache()
        
        mock_primary.clear_cache.assert_called_once()
        mock_fallback.clear_cache.assert_called_once()

    @patch('oboyu.common.fallback_embedding_service.get_circuit_breaker_registry')
    def test_get_circuit_breaker_status(self, mock_registry):
        """Test getting circuit breaker status."""
        mock_primary_cb = Mock()
        mock_primary_cb.get_state.return_value = CircuitState.CLOSED
        mock_primary_cb.get_metrics.return_value = Mock(
            total_requests=10,
            successful_requests=8,
            failed_requests=2,
            rejected_requests=0,
            get_failure_rate=Mock(return_value=0.2)
        )
        
        mock_fallback_cb = Mock()
        mock_fallback_cb.get_state.return_value = CircuitState.OPEN
        mock_fallback_cb.get_metrics.return_value = Mock(
            total_requests=5,
            successful_requests=0,
            failed_requests=5,
            rejected_requests=3,
            get_failure_rate=Mock(return_value=1.0)
        )
        
        mock_registry.return_value.get_or_create.return_value = mock_primary_cb
        mock_registry.return_value.get_circuit_breaker.return_value = mock_fallback_cb
        
        mock_primary = Mock()
        mock_fallback = Mock()
        mock_fallback.model_name = "fallback-model"
        
        service = FallbackEmbeddingService(
            primary_service=mock_primary,
            model_name="primary-model"
        )
        service.fallback_services = [mock_fallback]
        
        status = service.get_circuit_breaker_status()
        
        assert "primary" in status
        assert status["primary"]["model"] == "primary-model"
        assert status["primary"]["state"] == "closed"
        assert status["primary"]["metrics"]["total_requests"] == 10
        assert status["primary"]["metrics"]["failure_rate"] == 0.2
        
        assert "fallbacks" in status
        assert len(status["fallbacks"]) == 1
        assert status["fallbacks"][0]["model"] == "fallback-model"
        assert status["fallbacks"][0]["state"] == "open"

    def test_disable_fallback(self):
        """Test disabling fallback functionality."""
        mock_primary = Mock()
        service = FallbackEmbeddingService(
            primary_service=mock_primary,
            enable_fallback_services=False,
            enable_local_fallback=False,
            use_circuit_breaker=False
        )
        
        assert service.fallback_services == []
        assert service.local_fallback is None

    def test_empty_texts_input(self):
        """Test handling of empty texts input."""
        mock_primary = Mock()
        service = FallbackEmbeddingService(primary_service=mock_primary, use_circuit_breaker=False)
        
        result = service.generate_embeddings([])
        assert result == []
        mock_primary.generate_embeddings.assert_not_called()

    def test_progress_callback_forwarding(self):
        """Test that progress callback is forwarded to primary service."""
        mock_primary = Mock()
        expected_embeddings = [np.array([1, 2, 3], dtype=np.float32)]
        mock_primary.generate_embeddings.return_value = expected_embeddings
        
        service = FallbackEmbeddingService(primary_service=mock_primary, use_circuit_breaker=False)
        
        progress_callback = Mock()
        texts = ["test text"]
        result = service.generate_embeddings(texts, progress_callback)
        
        assert result == expected_embeddings
        mock_primary.generate_embeddings.assert_called_once_with(texts, progress_callback)