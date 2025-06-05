"""Factory for creating circuit breaker configurations from schema."""

from datetime import timedelta
from typing import Any, cast

from oboyu.common.circuit_breaker import CircuitBreakerConfig
from oboyu.config.schema import CircuitBreakerConfigSchema


def create_circuit_breaker_config(
    config_schema: CircuitBreakerConfigSchema | None = None,
    **overrides: Any,  # noqa: ANN401
) -> CircuitBreakerConfig:
    """Create a CircuitBreakerConfig from schema.
    
    Args:
        config_schema: Configuration schema to convert.
        **overrides: Additional parameter overrides.
        
    Returns:
        CircuitBreakerConfig instance.
        
    """
    if config_schema is None:
        config_schema = CircuitBreakerConfigSchema()
    
    # Convert schema to circuit breaker config parameters
    config_kwargs = {
        "enabled": config_schema.enabled,
        "failure_threshold": config_schema.failure_threshold,
        "recovery_timeout": timedelta(minutes=config_schema.recovery_timeout_minutes),
        "success_threshold": config_schema.success_threshold,
        "request_timeout": config_schema.request_timeout_seconds,
    }
    
    # Apply any overrides
    for key, value in overrides.items():
        if key in config_kwargs:
            config_kwargs[key] = value
    
    return CircuitBreakerConfig(
        failure_threshold=cast(int, config_kwargs["failure_threshold"]),
        recovery_timeout=timedelta(seconds=cast(float, config_kwargs["recovery_timeout"])),
        success_threshold=cast(int, config_kwargs["success_threshold"]),
        request_timeout=cast(float, config_kwargs["request_timeout"]),
        enabled=cast(bool, config_kwargs["enabled"]),
    )


def get_fallback_service_config(
    config_schema: CircuitBreakerConfigSchema | None = None,
) -> dict[str, Any]:
    """Get fallback service configuration from schema.
    
    Args:
        config_schema: Configuration schema to extract from.
        
    Returns:
        Dictionary with fallback service configuration.
        
    """
    if config_schema is None:
        config_schema = CircuitBreakerConfigSchema()
    
    return {
        "use_circuit_breaker": config_schema.enabled,
        "use_fallback": config_schema.enable_fallback_services,
        "fallback_model_names": config_schema.fallback_model_names,
    }


def should_enable_local_fallback(
    config_schema: CircuitBreakerConfigSchema | None = None,
) -> bool:
    """Check if local fallback should be enabled.
    
    Args:
        config_schema: Configuration schema to check.
        
    Returns:
        True if local fallback should be enabled.
        
    """
    if config_schema is None:
        config_schema = CircuitBreakerConfigSchema()
    
    return config_schema.enable_local_fallback
