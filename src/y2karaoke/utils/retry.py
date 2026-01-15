"""Retry utility with exponential backoff for external API calls."""

import time
from functools import wraps
from typing import Callable, TypeVar, Any, Optional, Type, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 30.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0


def retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry with (exception, attempt)

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        # Log the retry
                        logger.debug(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}"
                        )

                        # Call the on_retry callback if provided
                        if on_retry:
                            on_retry(e, attempt + 1)

                        # Wait before retrying
                        time.sleep(delay)

                        # Increase delay with backoff, but cap at max_delay
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.warning(
                            f"All {max_retries} retries exhausted for {func.__name__}: {e}"
                        )

            # If we've exhausted all retries, raise the last exception
            if last_exception:
                raise last_exception

            # This should never happen, but satisfies type checker
            raise RuntimeError("Unexpected state in retry logic")

        return wrapper
    return decorator


def retry_request(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    **kwargs: Any,
) -> T:
    """
    Execute a function with retry logic (non-decorator version).

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception: Optional[Exception] = None
    delay = base_delay

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                logger.debug(f"Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(delay)
                delay = min(delay * DEFAULT_BACKOFF_FACTOR, DEFAULT_MAX_DELAY)

    if last_exception:
        raise last_exception

    raise RuntimeError("Unexpected state in retry logic")
