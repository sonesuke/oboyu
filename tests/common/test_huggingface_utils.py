"""Tests for Hugging Face utilities."""

from __future__ import annotations

import socket
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.utils import EntryNotFoundError, OfflineModeIsEnabled

from oboyu.common.huggingface_utils import (
    HuggingFaceAuthenticationError,
    HuggingFaceError,
    HuggingFaceModelNotFoundError,
    HuggingFaceNetworkError,
    HuggingFaceRateLimitError,
    HuggingFaceTimeoutError,
    check_huggingface_connectivity,
    get_fallback_models,
    get_model_suggestions,
    get_user_friendly_error_message,
    safe_model_download,
    validate_model_exists,
)


class TestCheckHuggingFaceConnectivity:
    """Test connectivity checking functionality."""

    @patch("httpx.Client")
    def test_connectivity_success(self, mock_client_class: Mock) -> None:
        """Test successful connectivity check."""
        mock_response = Mock(status_code=200)
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client_class.return_value = mock_client

        assert check_huggingface_connectivity() is True

    @patch("httpx.Client")
    def test_connectivity_failure_network_error(self, mock_client_class: Mock) -> None:
        """Test connectivity check with network error."""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection failed")
        mock_client.__enter__.return_value = mock_client
        mock_client_class.return_value = mock_client

        assert check_huggingface_connectivity() is False

    @patch("httpx.Client")
    def test_connectivity_failure_timeout(self, mock_client_class: Mock) -> None:
        """Test connectivity check with timeout."""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("Timeout")
        mock_client.__enter__.return_value = mock_client
        mock_client_class.return_value = mock_client

        assert check_huggingface_connectivity() is False

    @patch("httpx.Client")
    def test_connectivity_failure_dns_error(self, mock_client_class: Mock) -> None:
        """Test connectivity check with DNS resolution error."""
        mock_client = MagicMock()
        mock_client.get.side_effect = socket.gaierror("DNS resolution failed")
        mock_client.__enter__.return_value = mock_client
        mock_client_class.return_value = mock_client

        assert check_huggingface_connectivity() is False

    @patch("httpx.Client")
    def test_connectivity_failure_unexpected_error(self, mock_client_class: Mock) -> None:
        """Test connectivity check with unexpected error."""
        mock_client = MagicMock()
        mock_client.get.side_effect = RuntimeError("Unexpected error")
        mock_client.__enter__.return_value = mock_client
        mock_client_class.return_value = mock_client

        assert check_huggingface_connectivity() is False


class TestSafeModelDownload:
    """Test safe model download functionality."""

    def test_download_success_first_try(self) -> None:
        """Test successful download on first attempt."""
        mock_func = Mock(return_value="model_data")
        result = safe_model_download("test/model", mock_func)
        assert result == "model_data"
        assert mock_func.call_count == 1

    def test_download_success_after_retry(self) -> None:
        """Test successful download after retries."""
        mock_func = Mock(side_effect=[httpx.TimeoutException("Timeout"), "model_data"])
        
        with patch("oboyu.common.huggingface_utils.check_huggingface_connectivity", return_value=True):
            with patch("time.sleep"):
                result = safe_model_download("test/model", mock_func, max_retries=2)
        
        assert result == "model_data"
        assert mock_func.call_count == 2

    def test_download_model_not_found(self) -> None:
        """Test download with model not found error."""
        mock_func = Mock(side_effect=RepositoryNotFoundError("Not found"))
        
        with pytest.raises(HuggingFaceModelNotFoundError) as exc_info:
            safe_model_download("test/model", mock_func)
        
        assert "Model 'test/model' not found" in str(exc_info.value)

    def test_download_authentication_error(self) -> None:
        """Test download with authentication error."""
        mock_response = Mock(status_code=401, headers={})
        mock_func = Mock(side_effect=HfHubHTTPError("Unauthorized", response=mock_response))
        
        with pytest.raises(HuggingFaceAuthenticationError) as exc_info:
            safe_model_download("test/model", mock_func)
        
        assert "Authentication required" in str(exc_info.value)

    def test_download_rate_limit_error(self) -> None:
        """Test download with rate limit error."""
        mock_response = Mock(status_code=429, headers={})
        mock_func = Mock(side_effect=HfHubHTTPError("Rate limited", response=mock_response))
        
        with pytest.raises(HuggingFaceRateLimitError) as exc_info:
            safe_model_download("test/model", mock_func, max_retries=0)
        
        assert "rate limit exceeded" in str(exc_info.value)

    def test_download_rate_limit_with_retry_after(self) -> None:
        """Test download with rate limit and retry-after header."""
        mock_response = Mock(status_code=429, headers={"retry-after": "5"})
        mock_func = Mock(
            side_effect=[
                HfHubHTTPError("Rate limited", response=mock_response),
                "model_data",
            ]
        )
        
        with patch("time.sleep") as mock_sleep:
            result = safe_model_download("test/model", mock_func, max_retries=1)
        
        assert result == "model_data"
        mock_sleep.assert_called_with(5.0)

    def test_download_network_error_no_connectivity(self) -> None:
        """Test download with network error when Hub is not accessible."""
        mock_func = Mock(side_effect=httpx.ConnectError("Connection failed"))
        
        with patch("oboyu.common.huggingface_utils.check_huggingface_connectivity", return_value=False):
            with pytest.raises(HuggingFaceNetworkError) as exc_info:
                safe_model_download("test/model", mock_func)
        
        assert "Cannot connect to Hugging Face Hub" in str(exc_info.value)

    def test_download_timeout_error_after_retries(self) -> None:
        """Test download timeout after all retries."""
        mock_func = Mock(side_effect=httpx.TimeoutException("Timeout"))
        
        with patch("oboyu.common.huggingface_utils.check_huggingface_connectivity", return_value=True):
            with patch("time.sleep"):
                with pytest.raises(HuggingFaceTimeoutError) as exc_info:
                    safe_model_download("test/model", mock_func, max_retries=2)
        
        assert "Timeout downloading model" in str(exc_info.value)

    def test_download_offline_mode_with_cache(self) -> None:
        """Test download in offline mode with cached model."""
        mock_func = Mock(side_effect=[OfflineModeIsEnabled(), "cached_model"])
        cache_dir = Path("/cache")
        
        with patch("oboyu.common.huggingface_utils._is_model_cached", return_value=True):
            result = safe_model_download(
                "test/model", mock_func, cache_dir=cache_dir
            )
        
        assert result == "cached_model"
        assert mock_func.call_count == 2

    def test_download_offline_mode_without_cache(self) -> None:
        """Test download in offline mode without cached model."""
        mock_func = Mock(side_effect=OfflineModeIsEnabled())
        cache_dir = Path("/cache")
        
        with patch("oboyu.common.huggingface_utils._is_model_cached", return_value=False):
            with pytest.raises(HuggingFaceNetworkError) as exc_info:
                safe_model_download("test/model", mock_func, cache_dir=cache_dir)
        
        assert "Cannot download model in offline mode" in str(exc_info.value)

    def test_download_entry_not_found_error(self) -> None:
        """Test download with missing file error."""
        mock_func = Mock(side_effect=EntryNotFoundError("File not found"))
        
        with pytest.raises(HuggingFaceModelNotFoundError) as exc_info:
            safe_model_download("test/model", mock_func)
        
        assert "File not found in model" in str(exc_info.value)


class TestUserFriendlyErrorMessages:
    """Test user-friendly error message generation."""

    def test_network_error_message(self) -> None:
        """Test network error message."""
        error = HuggingFaceNetworkError("Connection failed")
        message = get_user_friendly_error_message(error)
        
        assert "Cannot connect to Hugging Face Hub" in message
        assert "Internet connection" in message
        assert "Firewall" in message
        assert "cached models" in message

    def test_model_not_found_message(self) -> None:
        """Test model not found error message."""
        error = HuggingFaceModelNotFoundError("Model 'org/model' not found")
        message = get_user_friendly_error_message(error)
        
        assert "Model 'org/model' not found" in message
        assert "spelling is correct" in message
        assert "https://huggingface.co/org/model" in message
        assert "cl-nagoya/ruri-v3-30m" in message

    def test_rate_limit_message(self) -> None:
        """Test rate limit error message."""
        error = HuggingFaceRateLimitError("Rate limit exceeded")
        message = get_user_friendly_error_message(error)
        
        assert "rate limit reached" in message
        assert "15-30 minutes" in message
        assert "access token" in message
        assert "HF_TOKEN" in message

    def test_authentication_message(self) -> None:
        """Test authentication error message."""
        error = HuggingFaceAuthenticationError("Access denied")
        message = get_user_friendly_error_message(error)
        
        assert "Access denied" in message
        assert "private" in message
        assert "HF_TOKEN" in message
        assert "https://huggingface.co/settings/tokens" in message

    def test_timeout_message(self) -> None:
        """Test timeout error message."""
        error = HuggingFaceTimeoutError("Download timeout")
        message = get_user_friendly_error_message(error)
        
        assert "Download timeout" in message
        assert "Slow internet" in message
        assert "Large model size" in message
        assert "smaller model variant" in message

    def test_generic_error_message(self) -> None:
        """Test generic error message."""
        error = HuggingFaceError("Generic error")
        message = get_user_friendly_error_message(error)
        
        assert "Hugging Face Hub error" in message
        assert "Generic error" in message


class TestModelValidation:
    """Test model validation functionality."""

    @patch("oboyu.common.huggingface_utils.HfApi")
    def test_validate_model_exists_true(self, mock_api_class: Mock) -> None:
        """Test validation when model exists."""
        mock_api = Mock()
        mock_api.model_info.return_value = {"id": "test/model"}
        mock_api_class.return_value = mock_api
        
        assert validate_model_exists("test/model") is True

    @patch("oboyu.common.huggingface_utils.HfApi")
    def test_validate_model_exists_false(self, mock_api_class: Mock) -> None:
        """Test validation when model doesn't exist."""
        mock_api = Mock()
        mock_api.model_info.side_effect = RepositoryNotFoundError("Not found")
        mock_api_class.return_value = mock_api
        
        assert validate_model_exists("test/model") is False


class TestModelSuggestions:
    """Test model suggestion functionality."""

    @patch("oboyu.common.huggingface_utils.HfApi")
    def test_get_model_suggestions_success(self, mock_api_class: Mock) -> None:
        """Test getting model suggestions."""
        mock_model1 = Mock(id="org/model1")
        mock_model2 = Mock(id="org/model2")
        mock_api = Mock()
        mock_api.list_models.return_value = [mock_model1, mock_model2]
        mock_api_class.return_value = mock_api
        
        suggestions = get_model_suggestions("ruri", task="sentence-similarity")
        
        assert suggestions == ["org/model1", "org/model2"]
        mock_api.list_models.assert_called_once_with(
            search="ruri",
            task="sentence-similarity",
            limit=5,
            sort="downloads",
            direction=-1,
        )

    @patch("oboyu.common.huggingface_utils.HfApi")
    def test_get_model_suggestions_error(self, mock_api_class: Mock) -> None:
        """Test getting model suggestions with error."""
        mock_api = Mock()
        mock_api.list_models.side_effect = Exception("API error")
        mock_api_class.return_value = mock_api
        
        suggestions = get_model_suggestions("ruri")
        
        assert suggestions == []


class TestFallbackModels:
    """Test fallback model functionality."""

    def test_get_fallback_models_embedding(self) -> None:
        """Test getting embedding fallback models."""
        fallbacks = get_fallback_models("embedding")
        
        assert len(fallbacks) == 3
        assert fallbacks[0][0] == "cl-nagoya/ruri-v3-30m"
        assert "30M parameters" in fallbacks[0][1]

    def test_get_fallback_models_reranker(self) -> None:
        """Test getting reranker fallback models."""
        fallbacks = get_fallback_models("reranker")
        
        assert len(fallbacks) == 3
        assert fallbacks[0][0] == "cl-nagoya/ruri-reranker-small"
        assert "small" in fallbacks[0][1]

    def test_get_fallback_models_tokenizer(self) -> None:
        """Test getting tokenizer fallback models."""
        fallbacks = get_fallback_models("tokenizer")
        
        assert len(fallbacks) == 3
        assert fallbacks[0][0] == "cl-tohoku/bert-base-japanese-v3"
        assert "BERT" in fallbacks[0][1]

    def test_get_fallback_models_unknown_type(self) -> None:
        """Test getting fallback models for unknown type."""
        fallbacks = get_fallback_models("unknown")
        
        assert fallbacks == []


class TestHelperFunctions:
    """Test helper functions."""

    def test_is_model_cached_true(self, tmp_path: Path) -> None:
        """Test checking if model is cached."""
        cache_dir = tmp_path / "cache"
        model_dir = cache_dir / "models--org--model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").touch()
        
        from oboyu.common.huggingface_utils import _is_model_cached
        
        assert _is_model_cached("org/model", cache_dir) is True

    def test_is_model_cached_false(self, tmp_path: Path) -> None:
        """Test checking if model is not cached."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        
        from oboyu.common.huggingface_utils import _is_model_cached
        
        assert _is_model_cached("org/model", cache_dir) is False

    def test_extract_model_id_from_error_found(self) -> None:
        """Test extracting model ID from error message."""
        error = HuggingFaceModelNotFoundError("Model 'org/model' not found")
        
        from oboyu.common.huggingface_utils import _extract_model_id_from_error
        
        assert _extract_model_id_from_error(error) == "org/model"

    def test_extract_model_id_from_error_not_found(self) -> None:
        """Test extracting model ID when not in error message."""
        error = HuggingFaceError("Generic error")
        
        from oboyu.common.huggingface_utils import _extract_model_id_from_error
        
        assert _extract_model_id_from_error(error) == "unknown"