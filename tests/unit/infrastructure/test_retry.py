"""Tests for retry utilities with exponential backoff."""

import pytest
from unittest.mock import patch, MagicMock
import time

from y2karaoke.utils.retry import (
    retry_with_backoff,
    RetryManager,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
)


class TestRetryWithBackoff:
    def test_success_on_first_attempt(self):
        """Function succeeds immediately, no retries needed."""
        call_count = 0

        @retry_with_backoff(max_retries=3)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed()
        assert result == "success"
        assert call_count == 1

    @patch("time.sleep")
    def test_success_after_retry(self, mock_sleep):
        """Function fails then succeeds on retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 2
        mock_sleep.assert_called_once()  # One retry delay

    @patch("time.sleep")
    def test_exhausts_retries_then_raises(self, mock_sleep):
        """Function fails all attempts, raises exception."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent error")

        with pytest.raises(ValueError, match="Permanent error"):
            always_fail()

        assert call_count == 3  # Initial + 2 retries
        assert mock_sleep.call_count == 2  # 2 delays

    @patch("time.sleep")
    def test_only_retries_specified_exceptions(self, mock_sleep):
        """Only retry on specified exception types."""
        call_count = 0

        @retry_with_backoff(max_retries=3, exceptions=(ValueError,))
        def raise_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Wrong type")

        with pytest.raises(TypeError):
            raise_type_error()

        assert call_count == 1  # No retries for TypeError
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_retries_specified_exception(self, mock_sleep):
        """Retry only on specified exception type."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.1, exceptions=(ValueError,))
        def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Expected error")

        with pytest.raises(ValueError):
            raise_value_error()

        assert call_count == 3  # Initial + 2 retries

    @patch("time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        """Delay increases exponentially."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=100.0)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Error")

        with pytest.raises(ValueError):
            always_fail()

        # Check delays are increasing (with jitter, just check order)
        calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert len(calls) == 3
        # First delay ~1s, second ~2s, third ~4s (plus jitter)
        assert calls[0] < calls[1] < calls[2]

    @patch("time.sleep")
    def test_max_delay_cap(self, mock_sleep):
        """Delay is capped at max_delay."""
        call_count = 0

        @retry_with_backoff(max_retries=10, base_delay=10.0, max_delay=5.0)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Error")

        with pytest.raises(ValueError):
            always_fail()

        # All delays should be capped at ~5.0 (plus small jitter)
        for call in mock_sleep.call_args_list:
            delay = call[0][0]
            assert delay <= 5.5  # max_delay + 10% jitter

    def test_preserves_function_metadata(self):
        """Decorated function preserves original name and docstring."""

        @retry_with_backoff()
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestRetryManager:
    def test_success_no_retry(self):
        """Context manager with no errors doesn't retry."""
        attempt_count = 0
        with RetryManager("test_op", max_retries=3) as rm:
            attempt_count += 1
        assert attempt_count == 1

    @patch("time.sleep")
    def test_exit_suppresses_retryable_exception(self, mock_sleep):
        """Context manager __exit__ suppresses exception when retries remain."""
        rm = RetryManager("test_op", max_retries=3, base_delay=0.1)
        rm.__enter__()

        # First attempt fails - should suppress and increment
        result = rm.__exit__(ValueError, ValueError("error"), None)
        assert result is True  # Exception suppressed
        assert rm.attempt == 1
        mock_sleep.assert_called_once()

    def test_exit_propagates_after_max_retries(self):
        """Context manager __exit__ propagates exception after max retries."""
        rm = RetryManager("test_op", max_retries=2)
        rm.__enter__()
        rm.attempt = 2  # Already at max

        # Should not suppress exception
        result = rm.__exit__(ValueError, ValueError("error"), None)
        assert result is False  # Exception propagates

    def test_exit_propagates_non_retryable_exception(self):
        """Context manager __exit__ propagates non-retryable exceptions."""
        rm = RetryManager("test_op", max_retries=3, exceptions=(ValueError,))
        rm.__enter__()

        # TypeError is not in exceptions list
        result = rm.__exit__(TypeError, TypeError("error"), None)
        assert result is False  # Exception propagates

    def test_init_parameters(self):
        """RetryManager stores configuration correctly."""
        rm = RetryManager(
            "my_operation",
            max_retries=5,
            base_delay=2.0,
            exceptions=(ValueError, TypeError),
        )
        assert rm.operation_name == "my_operation"
        assert rm.max_retries == 5
        assert rm.base_delay == 2.0
        assert rm.exceptions == (ValueError, TypeError)
        assert rm.attempt == 0

    def test_enter_returns_self(self):
        """__enter__ returns the manager instance."""
        rm = RetryManager("test")
        result = rm.__enter__()
        assert result is rm


class TestDefaultConstants:
    def test_default_max_retries(self):
        assert DEFAULT_MAX_RETRIES == 3

    def test_default_base_delay(self):
        assert DEFAULT_BASE_DELAY == 1.0

    def test_default_max_delay(self):
        assert DEFAULT_MAX_DELAY == 60.0
