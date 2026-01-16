"""Retry utilities with exponential backoff."""

import time
import random
from typing import Callable, Any, Type, Tuple
from functools import wraps

from ..utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0

def retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """Decorator for retrying functions with exponential backoff."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts")
                        raise
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    logger.info(f"Retrying in {total_delay:.1f}s...")
                    time.sleep(total_delay)
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            
            # This should never be reached, but just in case
            raise last_exception or Exception("Unknown error")
        
        return wrapper
    return decorator

class RetryManager:
    """Context manager for retry operations."""
    
    def __init__(
        self,
        operation_name: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.operation_name = operation_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.exceptions = exceptions
        self.attempt = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, self.exceptions):
            if self.attempt < self.max_retries:
                delay = self.base_delay * (2 ** self.attempt)
                logger.warning(f"Attempt {self.attempt + 1} failed for {self.operation_name}")
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
                self.attempt += 1
                return True  # Suppress the exception
        return False  # Let the exception propagate
