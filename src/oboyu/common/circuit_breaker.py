"""Circuit breaker pattern implementation for external service reliability."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"        # Normal operation - requests are allowed
    OPEN = "open"           # Failure state - requests are blocked
    HALF_OPEN = "half_open" # Testing state - limited requests allowed


class CircuitBreakerError(Exception):
    """Raised when circuit breaker blocks a request."""

    def __init__(self, message: str, circuit_name: str, failure_count: int) -> None:
        """Initialize circuit breaker error.
        
        Args:
            message: Error message.
            circuit_name: Name of the circuit breaker.
            failure_count: Current failure count.
            
        """
        super().__init__(message)
        self.circuit_name = circuit_name
        self.failure_count = failure_count


class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: timedelta = timedelta(minutes=1),
        success_threshold: int = 3,
        request_timeout: float = 30.0,
        enabled: bool = True,
    ) -> None:
        """Initialize circuit breaker configuration.
        
        Args:
            failure_threshold: Number of failures before opening circuit.
            recovery_timeout: Time to wait before trying half-open state.
            success_threshold: Consecutive successes needed to close circuit.
            request_timeout: Timeout for individual requests.
            enabled: Whether circuit breaker is enabled.
            
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.request_timeout = request_timeout
        self.enabled = enabled


class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker."""

    def __init__(self) -> None:
        """Initialize metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rejected_requests = 0
        self.last_failure_time: datetime | None = None
        self.last_success_time: datetime | None = None
        self.state_transitions: list[tuple[datetime, CircuitState]] = []

    def record_success(self) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_success_time = datetime.now()

    def record_failure(self) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_failure_time = datetime.now()

    def record_rejection(self) -> None:
        """Record a rejected request."""
        self.rejected_requests += 1

    def record_state_transition(self, new_state: CircuitState) -> None:
        """Record a state transition."""
        self.state_transitions.append((datetime.now(), new_state))

    def get_failure_rate(self) -> float:
        """Get current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


class CircuitBreaker(Generic[T]):
    """Abstract base class for circuit breakers."""

    def __init__(self, name: str, config: CircuitBreakerConfig) -> None:
        """Initialize circuit breaker.
        
        Args:
            name: Name identifier for this circuit breaker.
            config: Configuration for circuit breaker behavior.
            
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.last_attempt_time: datetime | None = None
        self.metrics = CircuitBreakerMetrics()
        self._lock = threading.RLock()

    def call(self, func: Callable[[], T], *args: Any, **kwargs: Any) -> T:  # noqa: ANN401
        """Execute a function through the circuit breaker.
        
        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
            
        Returns:
            Result of the function call.
            
        Raises:
            CircuitBreakerError: If circuit is open and request is blocked.
            
        """
        if not self.config.enabled:
            return func(*args, **kwargs)

        with self._lock:
            if self._should_reject_request():
                self.metrics.record_rejection()
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Failed {self.failure_count} times. "
                    f"Will retry after {self.config.recovery_timeout}.",
                    self.name,
                    self.failure_count,
                )

            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit breaker '{self.name}' testing request in half-open state")

        try:
            self.last_attempt_time = datetime.now()
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _should_reject_request(self) -> bool:
        """Check if the request should be rejected based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return False

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
                return False
            return True

        # HALF_OPEN state - allow request but monitor closely
        return False

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful request."""
        with self._lock:
            self.success_count += 1
            self.metrics.record_success()

            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
                    logger.info(
                        f"Circuit breaker '{self.name}' recovered after "
                        f"{self.success_count} successful requests"
                    )

    def _on_failure(self, error: Exception) -> None:
        """Handle failed request."""
        with self._lock:
            self.failure_count += 1
            self.success_count = 0  # Reset success count on any failure
            self.last_failure_time = datetime.now()
            self.metrics.record_failure()

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure {self.failure_count}: {error}"
            )

            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
                    logger.error(
                        f"Circuit breaker '{self.name}' opened after "
                        f"{self.failure_count} failures"
                    )
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state should open the circuit
                self._transition_to_open()
                logger.warning(
                    f"Circuit breaker '{self.name}' reopened due to failure in half-open state"
                )

    def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.metrics.record_state_transition(self.state)
        logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to closed")

    def _transition_to_open(self) -> None:
        """Transition circuit to open state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.metrics.record_state_transition(self.state)
        logger.warning(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to open")

    def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.metrics.record_state_transition(self.state)
        logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to half-open")

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self.metrics

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            old_state = self.state
            self._transition_to_closed()
            logger.info(f"Circuit breaker '{self.name}' manually reset from {old_state.value}")

    def force_open(self) -> None:
        """Manually force circuit breaker to open state."""
        with self._lock:
            old_state = self.state
            # Set failure time to ensure circuit stays open for recovery timeout
            self.last_failure_time = datetime.now()
            self._transition_to_open()
            logger.warning(f"Circuit breaker '{self.name}' manually forced open from {old_state.value}")


class HuggingFaceCircuitBreaker(CircuitBreaker[T]):
    """Circuit breaker specifically designed for HuggingFace Hub operations."""

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialize HuggingFace circuit breaker.
        
        Args:
            config: Configuration for circuit breaker. Uses defaults if None.
            
        """
        if config is None:
            config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=timedelta(minutes=1),
                success_threshold=3,
                request_timeout=30.0,
                enabled=True,
            )
        super().__init__("HuggingFace", config)

    def _should_trip_on_error(self, error: Exception) -> bool:
        """Determine if an error should count towards circuit breaker failures.
        
        Some errors (like authentication errors) shouldn't trip the circuit breaker
        as they indicate configuration issues, not service availability issues.
        
        Args:
            error: The exception that occurred.
            
        Returns:
            True if this error should count as a failure.
            
        """
        from oboyu.common.huggingface_utils import (
            HuggingFaceAuthenticationError,
            HuggingFaceModelNotFoundError,
            HuggingFaceNetworkError,
            HuggingFaceRateLimitError,
            HuggingFaceTimeoutError,
        )

        # Network, timeout, and rate limit errors should trip the circuit
        if isinstance(error, (HuggingFaceNetworkError, HuggingFaceTimeoutError, HuggingFaceRateLimitError)):
            return True

        # Authentication and model not found errors shouldn't trip the circuit
        # as they indicate configuration issues, not service unavailability
        if isinstance(error, (HuggingFaceAuthenticationError, HuggingFaceModelNotFoundError)):
            return False

        # For other errors, be conservative and trip the circuit
        return True

    def _on_failure(self, error: Exception) -> None:
        """Handle failed request with HuggingFace-specific logic."""
        if self._should_trip_on_error(error):
            super()._on_failure(error)
        else:
            # Still record the failure in metrics but don't count towards circuit state
            self.metrics.record_failure()
            logger.debug(
                f"Circuit breaker '{self.name}' ignoring error for circuit state: {error}"
            )


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self) -> None:
        """Initialize registry."""
        self._circuit_breakers: dict[str, CircuitBreaker[Any]] = {}
        self._lock = threading.RLock()

    def get_or_create(
        self,
        name: str,
        circuit_type: type[CircuitBreaker[T]] = HuggingFaceCircuitBreaker,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker[T]:
        """Get or create a circuit breaker.
        
        Args:
            name: Name of the circuit breaker.
            circuit_type: Type of circuit breaker to create.
            config: Configuration for the circuit breaker.
            
        Returns:
            The circuit breaker instance.
            
        """
        with self._lock:
            if name not in self._circuit_breakers:
                if circuit_type is HuggingFaceCircuitBreaker:
                    self._circuit_breakers[name] = circuit_type(config)  # type: ignore[arg-type]
                else:
                    self._circuit_breakers[name] = circuit_type(name, config or CircuitBreakerConfig())
            return self._circuit_breakers[name]  # type: ignore[return-value]

    def get_all_metrics(self) -> dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers.
        
        Returns:
            Dictionary mapping circuit breaker names to their metrics.
            
        """
        with self._lock:
            return {name: cb.get_metrics() for name, cb in self._circuit_breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for cb in self._circuit_breakers.values():
                cb.reset()

    def get_circuit_breaker(self, name: str) -> CircuitBreaker[Any] | None:
        """Get a circuit breaker by name.
        
        Args:
            name: Name of the circuit breaker.
            
        Returns:
            The circuit breaker or None if not found.
            
        """
        return self._circuit_breakers.get(name)


# Global registry instance
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry.
    
    Returns:
        The global circuit breaker registry.
        
    """
    return _global_registry


def with_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
    circuit_type: type[CircuitBreaker[T]] = HuggingFaceCircuitBreaker,
) -> Callable[[Callable[[], T]], Callable[[], T]]:
    """Wrap a function with a circuit breaker.
    
    Args:
        name: Name of the circuit breaker.
        config: Configuration for the circuit breaker.
        circuit_type: Type of circuit breaker to use.
        
    Returns:
        Decorator function.
        
    """
    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:  # noqa: ANN401
            registry = get_circuit_breaker_registry()
            circuit_breaker = registry.get_or_create(name, circuit_type, config)
            return circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator
