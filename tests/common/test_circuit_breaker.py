"""Tests for circuit breaker functionality."""

import pytest
import time
from datetime import timedelta
from unittest.mock import Mock, patch

from oboyu.common.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    HuggingFaceCircuitBreaker,
    get_circuit_breaker_registry,
    with_circuit_breaker,
)
from oboyu.common.huggingface_utils import (
    HuggingFaceAuthenticationError,
    HuggingFaceModelNotFoundError,
    HuggingFaceNetworkError,
    HuggingFaceRateLimitError,
    HuggingFaceTimeoutError,
)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == timedelta(minutes=1)
        assert config.success_threshold == 3
        assert config.request_timeout == 30.0
        assert config.enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=timedelta(minutes=5),
            success_threshold=5,
            request_timeout=60.0,
            enabled=False,
        )
        assert config.failure_threshold == 10
        assert config.recovery_timeout == timedelta(minutes=5)
        assert config.success_threshold == 5
        assert config.request_timeout == 60.0
        assert config.enabled is False


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Test initial circuit breaker state."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=timedelta(seconds=1))
        circuit_breaker = CircuitBreaker("test", config)
        
        assert circuit_breaker.get_state() == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0

    def test_successful_request(self):
        """Test successful request processing."""
        config = CircuitBreakerConfig(failure_threshold=3)
        circuit_breaker = CircuitBreaker("test", config)
        
        def successful_func():
            return "success"
        
        result = circuit_breaker.call(successful_func)
        assert result == "success"
        assert circuit_breaker.get_state() == CircuitState.CLOSED
        assert circuit_breaker.get_metrics().successful_requests == 1

    def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=timedelta(seconds=1))
        circuit_breaker = CircuitBreaker("test", config)
        
        def failing_func():
            raise Exception("Test failure")
        
        # First 2 failures should keep circuit closed
        for i in range(2):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_func)
            assert circuit_breaker.get_state() == CircuitState.CLOSED
        
        # 3rd failure should open circuit (threshold=3 means open after 3 failures)
        with pytest.raises(Exception):
            circuit_breaker.call(failing_func)
        assert circuit_breaker.get_state() == CircuitState.OPEN

    def test_circuit_rejects_when_open(self):
        """Test circuit rejects requests when open."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=timedelta(seconds=1))
        circuit_breaker = CircuitBreaker("test", config)
        
        def failing_func():
            raise Exception("Test failure")
        
        # Trigger failures to open circuit (threshold=2 means circuit opens after 2 failures)
        for _ in range(2):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_func)
        
        assert circuit_breaker.get_state() == CircuitState.OPEN
        
        # Should reject subsequent requests
        with pytest.raises(CircuitBreakerError) as exc_info:
            circuit_breaker.call(lambda: "should not execute")
        
        assert "Circuit breaker 'test' is open" in str(exc_info.value)
        # After circuit opens, the rejected count should be 1
        assert circuit_breaker.get_metrics().rejected_requests == 1

    def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=timedelta(milliseconds=100))
        circuit_breaker = CircuitBreaker("test", config)
        
        def failing_func():
            raise Exception("Test failure")
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_func)
        
        assert circuit_breaker.get_state() == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Next call should transition to half-open
        def successful_func():
            return "success"
        
        result = circuit_breaker.call(successful_func)
        assert result == "success"
        assert circuit_breaker.get_state() == CircuitState.HALF_OPEN

    def test_circuit_closes_after_successes_in_half_open(self):
        """Test circuit closes after successful requests in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=2, 
            recovery_timeout=timedelta(milliseconds=100),
            success_threshold=2
        )
        circuit_breaker = CircuitBreaker("test", config)
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Test failure")))
        
        # Wait and transition to half-open
        time.sleep(0.2)
        
        def successful_func():
            return "success"
        
        # First success should transition to half-open
        result = circuit_breaker.call(successful_func)
        assert result == "success"
        assert circuit_breaker.get_state() == CircuitState.HALF_OPEN
        
        # Second success should close the circuit
        result = circuit_breaker.call(successful_func)
        assert result == "success"
        assert circuit_breaker.get_state() == CircuitState.CLOSED

    def test_circuit_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open state."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=timedelta(milliseconds=100))
        circuit_breaker = CircuitBreaker("test", config)
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Test failure")))
        
        # Wait and transition to half-open
        time.sleep(0.2)
        
        # First success should transition to half-open
        result = circuit_breaker.call(lambda: "success")
        assert result == "success"
        assert circuit_breaker.get_state() == CircuitState.HALF_OPEN
        
        # Failure in half-open should reopen circuit
        with pytest.raises(Exception):
            circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Test failure")))
        assert circuit_breaker.get_state() == CircuitState.OPEN

    def test_disabled_circuit_breaker(self):
        """Test circuit breaker when disabled."""
        config = CircuitBreakerConfig(enabled=False)
        circuit_breaker = CircuitBreaker("test", config)
        
        def failing_func():
            raise Exception("Test failure")
        
        # Should not open circuit when disabled
        for _ in range(10):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_func)
        
        assert circuit_breaker.get_state() == CircuitState.CLOSED

    def test_manual_reset(self):
        """Test manual circuit breaker reset."""
        config = CircuitBreakerConfig(failure_threshold=2)
        circuit_breaker = CircuitBreaker("test", config)
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("Test failure")))
        
        assert circuit_breaker.get_state() == CircuitState.OPEN
        
        # Manual reset
        circuit_breaker.reset()
        assert circuit_breaker.get_state() == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_force_open(self):
        """Test manually forcing circuit breaker open."""
        config = CircuitBreakerConfig()
        circuit_breaker = CircuitBreaker("test", config)
        
        assert circuit_breaker.get_state() == CircuitState.CLOSED
        
        circuit_breaker.force_open()
        assert circuit_breaker.get_state() == CircuitState.OPEN
        
        # Should reject requests
        with pytest.raises(CircuitBreakerError):
            circuit_breaker.call(lambda: "should not execute")


class TestHuggingFaceCircuitBreaker:
    """Test HuggingFace-specific circuit breaker functionality."""

    def test_network_errors_trip_circuit(self):
        """Test that network errors trip the circuit breaker."""
        circuit_breaker = HuggingFaceCircuitBreaker()
        
        def network_error_func():
            raise HuggingFaceNetworkError("Network error", "Connection failed")
        
        # Network errors should trip circuit
        for _ in range(5):
            with pytest.raises(HuggingFaceNetworkError):
                circuit_breaker.call(network_error_func)
        
        # Should open circuit after threshold (5 failures) and reject subsequent requests
        with pytest.raises(CircuitBreakerError):
            circuit_breaker.call(network_error_func)
        assert circuit_breaker.get_state() == CircuitState.OPEN

    def test_timeout_errors_trip_circuit(self):
        """Test that timeout errors trip the circuit breaker."""
        circuit_breaker = HuggingFaceCircuitBreaker()
        
        def timeout_error_func():
            raise HuggingFaceTimeoutError("Timeout error", "Request timed out")
        
        # Timeout errors should trip circuit
        for _ in range(5):
            with pytest.raises(HuggingFaceTimeoutError):
                circuit_breaker.call(timeout_error_func)
        
        with pytest.raises(CircuitBreakerError):
            circuit_breaker.call(timeout_error_func)
        assert circuit_breaker.get_state() == CircuitState.OPEN

    def test_rate_limit_errors_trip_circuit(self):
        """Test that rate limit errors trip the circuit breaker."""
        circuit_breaker = HuggingFaceCircuitBreaker()
        
        def rate_limit_error_func():
            raise HuggingFaceRateLimitError("Rate limit error", "Too many requests")
        
        # Rate limit errors should trip circuit
        for _ in range(5):
            with pytest.raises(HuggingFaceRateLimitError):
                circuit_breaker.call(rate_limit_error_func)
        
        with pytest.raises(CircuitBreakerError):
            circuit_breaker.call(rate_limit_error_func)
        assert circuit_breaker.get_state() == CircuitState.OPEN

    def test_auth_errors_dont_trip_circuit(self):
        """Test that authentication errors don't trip the circuit breaker."""
        circuit_breaker = HuggingFaceCircuitBreaker()
        
        def auth_error_func():
            raise HuggingFaceAuthenticationError("Auth error", "Invalid token")
        
        # Authentication errors should NOT trip circuit
        for _ in range(10):
            with pytest.raises(HuggingFaceAuthenticationError):
                circuit_breaker.call(auth_error_func)
        
        # Circuit should remain closed
        assert circuit_breaker.get_state() == CircuitState.CLOSED

    def test_model_not_found_errors_dont_trip_circuit(self):
        """Test that model not found errors don't trip the circuit breaker."""
        circuit_breaker = HuggingFaceCircuitBreaker()
        
        def model_error_func():
            raise HuggingFaceModelNotFoundError("Model not found", "Model does not exist")
        
        # Model not found errors should NOT trip circuit
        for _ in range(10):
            with pytest.raises(HuggingFaceModelNotFoundError):
                circuit_breaker.call(model_error_func)
        
        # Circuit should remain closed
        assert circuit_breaker.get_state() == CircuitState.CLOSED


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry functionality."""

    def test_get_or_create(self):
        """Test getting or creating circuit breakers."""
        registry = get_circuit_breaker_registry()
        
        # First call should create
        cb1 = registry.get_or_create("test1")
        assert cb1 is not None
        assert cb1.name == "HuggingFace"  # Default type is HuggingFaceCircuitBreaker
        
        # Second call should return same instance
        cb2 = registry.get_or_create("test1")
        assert cb1 is cb2

    def test_get_circuit_breaker(self):
        """Test getting circuit breaker by name."""
        registry = get_circuit_breaker_registry()
        
        # Should return None for non-existent
        cb = registry.get_circuit_breaker("non-existent")
        assert cb is None
        
        # Should return circuit breaker after creation
        created_cb = registry.get_or_create("test")
        retrieved_cb = registry.get_circuit_breaker("test")
        assert created_cb is retrieved_cb

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = get_circuit_breaker_registry()
        
        # Create and open some circuit breakers
        cb1 = registry.get_or_create("test1")
        cb2 = registry.get_or_create("test2")
        
        cb1.force_open()
        cb2.force_open()
        
        assert cb1.get_state() == CircuitState.OPEN
        assert cb2.get_state() == CircuitState.OPEN
        
        # Reset all
        registry.reset_all()
        
        assert cb1.get_state() == CircuitState.CLOSED
        assert cb2.get_state() == CircuitState.CLOSED

    def test_get_all_metrics(self):
        """Test getting metrics for all circuit breakers."""
        registry = get_circuit_breaker_registry()
        
        # Create some circuit breakers and use them
        cb1 = registry.get_or_create("test1")
        cb2 = registry.get_or_create("test2")
        
        cb1.call(lambda: "success")
        try:
            cb2.call(lambda: (_ for _ in ()).throw(Exception("failure")))
        except Exception:
            pass
        
        metrics = registry.get_all_metrics()
        assert "test1" in metrics
        assert "test2" in metrics
        assert metrics["test1"].successful_requests == 1
        assert metrics["test2"].failed_requests == 1


class TestWithCircuitBreakerDecorator:
    """Test the with_circuit_breaker decorator."""

    def test_decorator_success(self):
        """Test decorator with successful function."""
        @with_circuit_breaker("test_decorator")
        def successful_func():
            return "success"
        
        result = successful_func()
        assert result == "success"

    def test_decorator_failure(self):
        """Test decorator with failing function."""
        config = CircuitBreakerConfig(failure_threshold=2)
        
        @with_circuit_breaker("test_decorator_fail", config)
        def failing_func():
            raise Exception("Test failure")
        
        # Should allow failures until threshold
        for _ in range(2):
            with pytest.raises(Exception):
                failing_func()
        
        # Should raise circuit breaker error after threshold
        with pytest.raises(Exception):
            failing_func()
        
        # Next call should be rejected
        with pytest.raises(CircuitBreakerError):
            failing_func()


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics functionality."""

    def test_metrics_recording(self):
        """Test that metrics are recorded correctly."""
        config = CircuitBreakerConfig()
        circuit_breaker = CircuitBreaker("test", config)
        
        metrics = circuit_breaker.get_metrics()
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.rejected_requests == 0
        
        # Record successful request
        circuit_breaker.call(lambda: "success")
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        
        # Record failed request
        try:
            circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("failure")))
        except Exception:
            pass
        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1

    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        config = CircuitBreakerConfig()
        circuit_breaker = CircuitBreaker("test", config)
        
        metrics = circuit_breaker.get_metrics()
        assert metrics.get_failure_rate() == 0.0
        
        # One success
        circuit_breaker.call(lambda: "success")
        assert metrics.get_failure_rate() == 0.0
        
        # One failure
        try:
            circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("failure")))
        except Exception:
            pass
        assert metrics.get_failure_rate() == 0.5
        
        # Another failure
        try:
            circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("failure")))
        except Exception:
            pass
        assert metrics.get_failure_rate() == 2/3

    def test_state_transitions_recorded(self):
        """Test that state transitions are recorded."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=timedelta(milliseconds=100))
        circuit_breaker = CircuitBreaker("test", config)
        
        metrics = circuit_breaker.get_metrics()
        initial_transitions = len(metrics.state_transitions)
        
        # Open circuit
        for _ in range(3):
            try:
                circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("failure")))
            except Exception:
                pass
        
        # Should record transition to OPEN
        assert len(metrics.state_transitions) == initial_transitions + 1
        assert metrics.state_transitions[-1][1] == CircuitState.OPEN
        
        # Wait and trigger half-open
        time.sleep(0.2)
        circuit_breaker.call(lambda: "success")
        
        # Should record transition to HALF_OPEN
        assert len(metrics.state_transitions) == initial_transitions + 2
        assert metrics.state_transitions[-1][1] == CircuitState.HALF_OPEN