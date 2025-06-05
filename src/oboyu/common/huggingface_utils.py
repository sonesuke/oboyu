"""Utilities for handling Hugging Face Hub connectivity and errors."""

from __future__ import annotations

import logging
import socket
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    HfHubHTTPError,
    RepositoryNotFoundError,
)

# Import HuggingFace utils with fallback for older versions
EntryNotFoundError: type[Exception]
LocalEntryNotFoundError: type[Exception]
OfflineModeIsEnabled: type[Exception]

try:
    from huggingface_hub.utils import (
        EntryNotFoundError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
    )
except ImportError:
    # Fallback for older versions of huggingface_hub
    class EntryNotFoundError(Exception):  # type: ignore[misc,no-redef]
        """Fallback exception for entry not found errors."""

        pass
    
    class LocalEntryNotFoundError(Exception):  # type: ignore[misc,no-redef]
        """Fallback exception for local entry not found errors."""

        pass
    
    class OfflineModeIsEnabled(Exception):  # type: ignore[misc,no-redef]
        """Fallback exception for offline mode errors."""

        pass

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar("T")


class HuggingFaceError(Exception):
    """Base exception for Hugging Face related errors."""

    def __init__(self, message: str, technical_details: str | None = None) -> None:
        """Initialize the error with a user-friendly message and optional technical details."""
        super().__init__(message)
        self.message = message
        self.technical_details = technical_details


class HuggingFaceNetworkError(HuggingFaceError):
    """Raised when network connectivity issues prevent accessing Hugging Face Hub."""


class HuggingFaceModelNotFoundError(HuggingFaceError):
    """Raised when a requested model cannot be found on Hugging Face Hub."""


class HuggingFaceRateLimitError(HuggingFaceError):
    """Raised when Hugging Face API rate limits are exceeded."""


class HuggingFaceAuthenticationError(HuggingFaceError):
    """Raised when authentication is required to access a model."""


class HuggingFaceTimeoutError(HuggingFaceError):
    """Raised when operations timeout while accessing Hugging Face Hub."""


def check_huggingface_connectivity(timeout: float = 5.0) -> bool:
    """Check if Hugging Face Hub is accessible.

    Args:
        timeout: Maximum time to wait for response in seconds.

    Returns:
        True if Hub is accessible, False otherwise.

    """
    try:
        with httpx.Client() as client:
            response = client.get(
                "https://huggingface.co/api/models",
                timeout=timeout,
                params={"limit": 1},
            )
            return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, socket.gaierror):
        return False
    except Exception:
        logger.exception("Unexpected error checking Hugging Face connectivity")
        return False


def safe_model_download(
    model_id: str,
    download_func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    cache_dir: Path | None = None,
    use_circuit_breaker: bool = True,
) -> T:
    """Safely download a model with retry logic and error handling.

    Args:
        model_id: The model identifier (e.g., 'cl-nagoya/ruri-v3-30m').
        download_func: Function that performs the actual download.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        backoff_factor: Factor by which to multiply delay after each retry.
        cache_dir: Directory where models are cached.
        use_circuit_breaker: Whether to use circuit breaker for reliability.

    Returns:
        The result of the download function.

    Raises:
        HuggingFaceError: Various subclasses depending on the error type.

    """
    # Use circuit breaker if enabled
    if use_circuit_breaker:
        from oboyu.common.circuit_breaker import (
            CircuitBreakerError,
            get_circuit_breaker_registry,
        )
        
        registry = get_circuit_breaker_registry()
        circuit_breaker: Any = registry.get_or_create(f"huggingface_download_{model_id}")
        
        try:
            return circuit_breaker.call(
                lambda: _safe_model_download_impl(
                    model_id, download_func, max_retries, initial_delay, backoff_factor, cache_dir
                )
            )
        except CircuitBreakerError as e:
            # Convert circuit breaker error to appropriate HuggingFace error
            raise HuggingFaceNetworkError(
                message=f"Circuit breaker is open for model '{model_id}' due to repeated failures",
                technical_details=str(e),
            ) from e
    else:
        return _safe_model_download_impl(
            model_id, download_func, max_retries, initial_delay, backoff_factor, cache_dir
        )


def _safe_model_download_impl(
    model_id: str,
    download_func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    cache_dir: Path | None = None,
) -> T:
    """Implement safe model download without circuit breaker."""
    last_error: Exception | None = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{max_retries} for model {model_id}")
                time.sleep(delay)
                delay *= backoff_factor

            return download_func()

        except OfflineModeIsEnabled:
            # Check if model is available in cache
            if cache_dir and _is_model_cached(model_id, cache_dir):
                logger.info(f"Using cached model {model_id} in offline mode")
                return download_func()
            raise HuggingFaceNetworkError(
                message="Cannot download model in offline mode",
                technical_details="Hugging Face Hub is set to offline mode and model is not cached",
            )

        except RepositoryNotFoundError as e:
            raise HuggingFaceModelNotFoundError(
                message=f"Model '{model_id}' not found on Hugging Face Hub",
                technical_details=str(e),
            ) from e

        except EntryNotFoundError as e:
            raise HuggingFaceModelNotFoundError(
                message=f"File not found in model '{model_id}'",
                technical_details=str(e),
            ) from e

        except HfHubHTTPError as e:
            if e.response and e.response.status_code == 401:
                raise HuggingFaceAuthenticationError(
                    message=f"Authentication required to access model '{model_id}'",
                    technical_details="This model may be private. Please set HF_TOKEN environment variable.",
                ) from e
            elif e.response and e.response.status_code == 429:
                if attempt < max_retries:
                    # Extract retry-after header if available
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after:
                        delay = float(retry_after)
                    logger.warning(f"Rate limit hit, waiting {delay} seconds")
                    last_error = e
                    continue
                raise HuggingFaceRateLimitError(
                    message="Hugging Face API rate limit exceeded",
                    technical_details="Please wait 15-30 minutes before retrying or use an access token for higher limits.",
                ) from e
            else:
                last_error = e

        except (httpx.ConnectError, httpx.TimeoutException, socket.gaierror) as e:
            if not check_huggingface_connectivity():
                raise HuggingFaceNetworkError(
                    message="Cannot connect to Hugging Face Hub",
                    technical_details="Please check your internet connection and firewall settings.",
                ) from e
            last_error = e

        except (EntryNotFoundError, LocalEntryNotFoundError) as e:
            # These can occur when a specific file within a model repo is not found
            raise HuggingFaceModelNotFoundError(
                message=f"Required files not found for model '{model_id}'",
                technical_details=str(e),
            ) from e

        except HuggingFaceError:
            # Re-raise Hugging Face errors directly (already properly typed)
            raise
            
        except Exception as e:
            logger.exception(f"Unexpected error downloading model {model_id}")
            last_error = e

    # If we exhausted all retries
    if isinstance(last_error, (httpx.TimeoutException, TimeoutError)):
        raise HuggingFaceTimeoutError(
            message=f"Timeout downloading model '{model_id}'",
            technical_details="The download is taking too long. This may be due to slow connection or large model size.",
        ) from last_error
    else:
        raise HuggingFaceNetworkError(
            message=f"Failed to download model '{model_id}' after {max_retries} retries",
            technical_details=str(last_error) if last_error else None,
        ) from last_error


def get_user_friendly_error_message(error: HuggingFaceError) -> str:
    """Convert a HuggingFaceError to a user-friendly message with guidance.

    Args:
        error: The error to convert.

    Returns:
        A formatted string with clear guidance for the user.

    """
    if isinstance(error, HuggingFaceNetworkError):
        return (
            "❌ Cannot connect to Hugging Face Hub\n\n"
            "Please check:\n"
            "• Internet connection is working\n"
            "• Firewall is not blocking huggingface.co\n"
            "• VPN settings (try disabling temporarily)\n"
            "• Try again in a few minutes\n\n"
            "The system will attempt to use cached models if available."
        )

    elif isinstance(error, HuggingFaceModelNotFoundError):
        model_id = _extract_model_id_from_error(error)
        return (
            f"❌ Model '{model_id}' not found\n\n"
            "Please check:\n"
            f"• Model name spelling is correct\n"
            f"• Model exists at: https://huggingface.co/{model_id}\n"
            "• Organization and repository names are correct\n\n"
            "Recommended Japanese models:\n"
            "• cl-nagoya/ruri-v3-30m (embeddings)\n"
            "• cl-nagoya/ruri-reranker-small (reranking)"
        )

    elif isinstance(error, HuggingFaceRateLimitError):
        return (
            "⏰ Hugging Face rate limit reached\n\n"
            "Please wait 15-30 minutes before retrying.\n"
            "Consider using a Hugging Face access token for higher limits:\n"
            "• Create token at: https://huggingface.co/settings/tokens\n"
            "• Set environment variable: export HF_TOKEN=your_token"
        )

    elif isinstance(error, HuggingFaceAuthenticationError):
        return (
            "🔒 Access denied to model\n\n"
            "This model may be private and require authentication.\n"
            "Please set up a Hugging Face access token:\n"
            "• Create token at: https://huggingface.co/settings/tokens\n"
            "• Set environment variable: export HF_TOKEN=your_token\n"
            "• Ensure you have access to the private model"
        )

    elif isinstance(error, HuggingFaceTimeoutError):
        return (
            "⏱️ Download timeout\n\n"
            "The model download is taking too long. This could be due to:\n"
            "• Slow internet connection\n"
            "• Large model size\n"
            "• Temporary server issues\n\n"
            "Please try:\n"
            "• Check your internet speed\n"
            "• Try again during off-peak hours\n"
            "• Use a smaller model variant if available"
        )

    else:
        return (
            "❌ Hugging Face Hub error\n\n"
            f"Error: {error.message}\n\n"
            "Please try again or check the logs for more details."
        )


def validate_model_exists(model_id: str) -> bool:
    """Check if a model exists on Hugging Face Hub.

    Args:
        model_id: The model identifier to check.

    Returns:
        True if the model exists, False otherwise.

    """
    try:
        api = HfApi()
        info = api.model_info(model_id)
        return info is not None
    except Exception:
        return False


def get_model_suggestions(partial_name: str, task: str | None = None) -> list[str]:
    """Get model suggestions based on partial name and optional task.

    Args:
        partial_name: Partial model name to search for.
        task: Optional task filter (e.g., 'sentence-similarity', 'text-classification').

    Returns:
        List of suggested model IDs.

    """
    try:
        api = HfApi()
        models = api.list_models(
            search=partial_name,
            task=task,
            limit=5,
            sort="downloads",
            direction=-1,
        )
        return [model.id for model in models]
    except Exception:
        logger.exception("Failed to get model suggestions")
        return []


def _is_model_cached(model_id: str, cache_dir: Path) -> bool:
    """Check if a model is available in the local cache.

    Args:
        model_id: The model identifier.
        cache_dir: The cache directory to check.

    Returns:
        True if the model appears to be cached, False otherwise.

    """
    # This is a simplified check - actual caching structure may vary
    model_path = cache_dir / f"models--{model_id.replace('/', '--')}"
    return model_path.exists() and any(model_path.iterdir())


def _extract_model_id_from_error(error: HuggingFaceError) -> str:
    """Extract model ID from error message if possible.

    Args:
        error: The error to extract from.

    Returns:
        The model ID or 'unknown' if not found.

    """
    import re

    # Try to extract model ID from error message
    if match := re.search(r"'([^']+/[^']+)'", error.message):
        return match.group(1)
    return "unknown"


def with_huggingface_circuit_breaker(
    operation_name: str,
    use_circuit_breaker: bool = True,
) -> Callable[[Callable[[], T]], T]:
    """Execute a function with HuggingFace circuit breaker protection.
    
    Args:
        operation_name: Name of the operation for circuit breaker identification.
        use_circuit_breaker: Whether to use circuit breaker protection.
        
    Returns:
        Function result or raises appropriate HuggingFace error.
        
    Raises:
        HuggingFaceError: Various subclasses depending on the error type.
        
    """
    def wrapper(func: Callable[[], T]) -> T:
        if not use_circuit_breaker:
            return func()
            
        from oboyu.common.circuit_breaker import (
            CircuitBreakerError,
            get_circuit_breaker_registry,
        )
        
        registry = get_circuit_breaker_registry()
        circuit_breaker: Any = registry.get_or_create(f"huggingface_{operation_name}")
        
        try:
            return circuit_breaker.call(func)
        except CircuitBreakerError as e:
            # Convert circuit breaker error to appropriate HuggingFace error
            raise HuggingFaceNetworkError(
                message=f"Circuit breaker is open for '{operation_name}' due to repeated failures",
                technical_details=str(e),
            ) from e
    
    return wrapper


def get_fallback_models(model_type: str) -> list[tuple[str, str]]:
    """Get fallback model suggestions based on model type.

    Args:
        model_type: Type of model ('embedding', 'reranker', 'tokenizer').

    Returns:
        List of (model_id, description) tuples.

    """
    fallbacks = {
        "embedding": [
            ("cl-nagoya/ruri-v3-30m", "Japanese text embeddings (30M parameters)"),
            ("cl-nagoya/ruri-v3-base", "Japanese text embeddings (base size)"),
            ("sentence-transformers/all-MiniLM-L6-v2", "Multilingual embeddings"),
        ],
        "reranker": [
            ("cl-nagoya/ruri-reranker-small", "Japanese text reranking (small)"),
            ("cl-nagoya/ruri-reranker-base", "Japanese text reranking (base)"),
            ("BAAI/bge-reranker-base", "Multilingual reranking"),
        ],
        "tokenizer": [
            ("cl-tohoku/bert-base-japanese-v3", "Japanese BERT tokenizer"),
            ("line-corporation/line-distilbert-base-japanese", "Japanese DistilBERT"),
            ("bert-base-multilingual-cased", "Multilingual BERT tokenizer"),
        ],
    }
    return fallbacks.get(model_type, [])

