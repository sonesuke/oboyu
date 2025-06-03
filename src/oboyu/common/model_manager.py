"""Unified model management with lazy loading and caching."""
import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

from oboyu.common.huggingface_utils import (
    HuggingFaceError,
    get_user_friendly_error_message,
    safe_model_download,
)
from oboyu.common.paths import EMBEDDING_CACHE_DIR

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading models in the same process
_MODEL_CACHE: Dict[str, Any] = {}


class ModelManager(ABC):
    """Abstract base class for unified model management with lazy loading."""

    def __init__(
        self,
        model_name: str,
        model_type: str,
        use_onnx: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the model manager.

        Args:
            model_name: Name of the model to load
            model_type: Type of model (embedding, reranker)
            use_onnx: Whether to use ONNX optimization
            cache_dir: Directory to cache models (defaults to XDG cache path)
            **kwargs: Additional model-specific configuration

        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = "cpu"  # Always CPU-only
        self.use_onnx = use_onnx
        self.cache_dir = Path(cache_dir) if cache_dir else EMBEDDING_CACHE_DIR / "models"
        self.config = kwargs

        # Model instance (lazy loaded)
        self._model: Optional[Any] = None

        # Generate cache key for model instance caching
        self._cache_key = self._generate_cache_key()

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key for this model configuration."""
        # Include all relevant configuration in cache key
        config_str = f"{self.model_name}_{self.model_type}_{self.device}_{self.use_onnx}"

        # Add relevant config parameters
        for key, value in sorted(self.config.items()):
            if key in ["batch_size", "max_length", "max_seq_length", "quantization_config", "optimization_level"]:
                config_str += f"_{key}_{value}"

        # Hash to keep cache key reasonable length
        return hashlib.sha256(config_str.encode()).hexdigest()  # Use SHA256 instead of MD5

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Get the model, loading it lazily if not already loaded."""
        if self._model is None:
            # Check global cache first
            if self._cache_key in _MODEL_CACHE:
                self._model = _MODEL_CACHE[self._cache_key]
                logger.debug(f"Retrieved {self.model_type} model from cache: {self.model_name}")
            else:
                # Load model and cache it
                logger.debug(f"Loading {self.model_type} model: {self.model_name} (ONNX: {self.use_onnx})")
                self._model = self._load_model()
                _MODEL_CACHE[self._cache_key] = self._model
                logger.debug(f"Cached {self.model_type} model: {self.model_name}")

        return self._model

    @abstractmethod
    def _load_model(self) -> Any:  # noqa: ANN401
        """Load the model implementation.

        Returns:
            The loaded model instance

        """
        pass

    def get_cache_dir(self) -> Path:
        """Get the cache directory for this model type."""
        model_cache_dir = self.cache_dir / self.model_type / self.model_name.replace("/", "_")
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        return model_cache_dir

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the global model cache."""
        global _MODEL_CACHE
        _MODEL_CACHE.clear()
        logger.debug("Cleared global model cache")


class ONNXModelCache:
    """Handles ONNX model caching and conversion."""

    @staticmethod
    def get_onnx_path(
        model_name: str,
        model_type: str,
        cache_dir: Optional[Path] = None,
        quantized: bool = True,
    ) -> Path:
        """Get ONNX model path.

        Args:
            model_name: Name of the model
            model_type: Type of model (embedding, reranker)
            cache_dir: Cache directory (defaults to XDG cache)
            quantized: Whether to prefer quantized version

        Returns:
            Path to ONNX model file

        """
        if cache_dir is None:
            cache_dir = EMBEDDING_CACHE_DIR / "models"

        model_dir = cache_dir / "onnx" / model_name.replace("/", "_")

        # Try quantized model first if requested
        if quantized:
            onnx_path = model_dir / "model_quantized.onnx"
            if onnx_path.exists() and onnx_path.stat().st_size > 0:
                return onnx_path

        # Try optimized model
        onnx_path = model_dir / "model_optimized.onnx"
        if onnx_path.exists() and onnx_path.stat().st_size > 0:
            return onnx_path

        # Try basic model
        onnx_path = model_dir / "model.onnx"
        if onnx_path.exists() and onnx_path.stat().st_size > 0:
            return onnx_path

        raise FileNotFoundError(f"No ONNX model found for {model_name} in {model_dir}")

    @staticmethod
    def convert_to_onnx(
        model_name: str,
        model_type: str,
        cache_dir: Optional[Path] = None,
        apply_quantization: bool = True,
        quantization_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Convert model to ONNX format with error handling.

        Args:
            model_name: Name of the model to convert
            model_type: Type of model (embedding, reranker)
            cache_dir: Cache directory (defaults to XDG cache)
            apply_quantization: Whether to apply quantization
            quantization_config: Quantization configuration

        Returns:
            Path to converted ONNX model

        """
        if cache_dir is None:
            cache_dir = EMBEDDING_CACHE_DIR / "models"

        # Validate model type before proceeding
        if model_type not in ["embedding", "reranker"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_dir = cache_dir / "onnx" / model_name.replace("/", "_")

        def convert_model() -> Path:
            if model_type == "embedding":
                from oboyu.common.onnx.embedding_model import convert_to_onnx

                return convert_to_onnx(
                    model_name,
                    model_dir,
                    apply_quantization=apply_quantization,
                    quantization_config=quantization_config,
                )
            elif model_type == "reranker":
                from oboyu.common.onnx.cross_encoder_model import convert_cross_encoder_to_onnx

                return convert_cross_encoder_to_onnx(
                    model_name,
                    model_dir,
                    apply_quantization=apply_quantization,
                    quantization_config=quantization_config,
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        try:
            return safe_model_download(
                model_name,
                convert_model,
                cache_dir=cache_dir,
            )
        except HuggingFaceError:
            # Re-raise HuggingFace errors as-is for consistent handling
            raise
        except Exception as e:
            logger.exception(f"Failed to convert {model_type} model {model_name} to ONNX")
            raise RuntimeError(f"ONNX conversion failed for model '{model_name}': {e}") from e

    @staticmethod
    def get_or_convert_onnx_model(
        model_name: str,
        model_type: str,
        cache_dir: Optional[Path] = None,
        apply_quantization: bool = True,
        quantization_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Get ONNX model path, converting if necessary.

        Args:
            model_name: Name of the model
            model_type: Type of model (embedding, reranker)
            cache_dir: Cache directory (defaults to XDG cache)
            apply_quantization: Whether to apply quantization
            quantization_config: Quantization configuration

        Returns:
            Path to ONNX model file

        """
        try:
            return ONNXModelCache.get_onnx_path(model_name, model_type, cache_dir, quantized=apply_quantization)
        except FileNotFoundError:
            logger.debug(f"ONNX model not found, converting {model_name}...")
            return ONNXModelCache.convert_to_onnx(
                model_name,
                model_type,
                cache_dir,
                apply_quantization=apply_quantization,
                quantization_config=quantization_config,
            )


class EmbeddingModelManager(ModelManager):
    """Model manager for embedding models."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-30m",
        use_onnx: bool = True,
        max_seq_length: int = 8192,
        cache_dir: Optional[Union[str, Path]] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
        optimization_level: str = "none",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize embedding model manager.

        Args:
            model_name: Name of the embedding model
            use_onnx: Whether to use ONNX optimization
            max_seq_length: Maximum sequence length
            cache_dir: Directory to cache models
            quantization_config: ONNX quantization configuration
            optimization_level: ONNX optimization level
            **kwargs: Additional configuration

        """
        super().__init__(
            model_name=model_name,
            model_type="embedding",
            use_onnx=use_onnx,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            quantization_config=quantization_config,
            optimization_level=optimization_level,
            **kwargs,
        )
        self.max_seq_length = max_seq_length
        self.quantization_config = quantization_config or {"enabled": True, "weight_type": "uint8"}
        self.optimization_level = optimization_level

    def _load_model(self) -> Any:  # noqa: ANN401
        """Load embedding model (ONNX or PyTorch) with error handling."""
        try:
            if self.use_onnx:
                from oboyu.common.onnx.embedding_model import ONNXEmbeddingModel

                # Get or convert ONNX model with safe download
                def download_and_convert() -> Path:
                    return ONNXModelCache.get_or_convert_onnx_model(
                        self.model_name,
                        "embedding",
                        self.cache_dir,
                        apply_quantization=self.quantization_config.get("enabled", True),
                        quantization_config=self.quantization_config,
                    )

                onnx_path = safe_model_download(
                    self.model_name,
                    download_and_convert,
                    cache_dir=self.cache_dir,
                )

                return ONNXEmbeddingModel(
                    onnx_path,
                    max_seq_length=self.max_seq_length,
                    optimization_level=self.optimization_level,
                )
            else:
                # Load PyTorch model with safe download
                try:
                    from sentence_transformers import SentenceTransformer

                    # Silence SentenceTransformer logging
                    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
                except ImportError as e:
                    raise ImportError("sentence_transformers is required for embedding generation. Install with: pip install sentence_transformers") from e

                model_cache_dir = self.get_cache_dir()

                def download_model() -> SentenceTransformer:
                    return SentenceTransformer(
                        self.model_name,
                        device="cpu",  # Fixed to CPU only
                        cache_folder=str(model_cache_dir),
                    )

                model = safe_model_download(
                    self.model_name,
                    download_model,
                    cache_dir=self.cache_dir,
                )
                model.max_seq_length = self.max_seq_length
                return model

        except HuggingFaceError as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            # Re-raise with context for higher-level error handling
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}'. "
                f"Error: {get_user_friendly_error_message(e)}"
            ) from e
        except Exception as e:
            logger.exception(f"Unexpected error loading embedding model {self.model_name}")
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}") from e

    def get_dimensions(self) -> int:
        """Get the embedding dimensions."""
        return int(self.model.get_sentence_embedding_dimension())


class RerankerModelManager(ModelManager):
    """Model manager for reranker models."""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-reranker-small",
        use_onnx: bool = True,
        max_length: int = 512,
        cache_dir: Optional[Union[str, Path]] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
        optimization_level: str = "none",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize reranker model manager.

        Args:
            model_name: Name of the reranker model
            use_onnx: Whether to use ONNX optimization
            max_length: Maximum sequence length
            cache_dir: Directory to cache models
            quantization_config: ONNX quantization configuration
            optimization_level: ONNX optimization level
            **kwargs: Additional configuration

        """
        super().__init__(
            model_name=model_name,
            model_type="reranker",
            use_onnx=use_onnx,
            cache_dir=cache_dir,
            max_length=max_length,
            quantization_config=quantization_config,
            optimization_level=optimization_level,
            **kwargs,
        )
        self.max_length = max_length
        self.quantization_config = quantization_config or {"enabled": True, "weight_type": "uint8"}
        self.optimization_level = optimization_level

    def _load_model(self) -> Any:  # noqa: ANN401
        """Load reranker model (ONNX or PyTorch) with error handling."""
        try:
            if self.use_onnx:
                from oboyu.common.onnx.cross_encoder_model import ONNXCrossEncoderModel

                # Get or convert ONNX model with safe download
                def download_and_convert() -> Path:
                    return ONNXModelCache.get_or_convert_onnx_model(
                        self.model_name,
                        "reranker",
                        self.cache_dir,
                        apply_quantization=self.quantization_config.get("enabled", True),
                        quantization_config=self.quantization_config,
                    )

                onnx_path = safe_model_download(
                    self.model_name,
                    download_and_convert,
                    cache_dir=self.cache_dir,
                )

                return ONNXCrossEncoderModel(
                    onnx_path,
                    max_seq_length=self.max_length,
                    optimization_level=self.optimization_level,
                )
            else:
                # Load PyTorch model with safe download
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as e:
                    raise ImportError("sentence_transformers is required for reranking. Install with: pip install sentence_transformers") from e

                def download_model() -> CrossEncoder:
                    return CrossEncoder(
                        self.model_name,
                        device="cpu",  # Fixed to CPU only
                        max_length=self.max_length,
                        trust_remote_code=True,
                    )

                result = safe_model_download(
                    self.model_name,
                    download_model,
                    cache_dir=self.cache_dir,
                )
                return result  # type: ignore[no-any-return]

        except HuggingFaceError as e:
            logger.error(f"Failed to load reranker model {self.model_name}: {e}")
            # Re-raise with context for higher-level error handling
            raise RuntimeError(
                f"Failed to load reranker model '{self.model_name}'. "
                f"Error: {get_user_friendly_error_message(e)}"
            ) from e
        except Exception as e:
            logger.exception(f"Unexpected error loading reranker model {self.model_name}")
            raise RuntimeError(f"Failed to load reranker model '{self.model_name}': {e}") from e

def create_model_manager(
    model_type: str,
    model_name: str,
    use_onnx: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,  # noqa: ANN401
) -> ModelManager:
    """Create appropriate model manager instance.

    Args:
        model_type: Type of model (embedding, reranker)
        model_name: Name of the model
        use_onnx: Whether to use ONNX optimization
        cache_dir: Directory to cache models
        **kwargs: Additional model-specific configuration
    Returns:
        Model manager instance
    Raises:
        ValueError: If model_type is not supported

    """
    if model_type == "embedding":
        return EmbeddingModelManager(
            model_name=model_name,
            use_onnx=use_onnx,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif model_type == "reranker":
        return RerankerModelManager(
            model_name=model_name,
            use_onnx=use_onnx,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Must be 'embedding' or 'reranker'")
