"""Performance monitoring utilities."""

import time
import functools
from typing import Callable, Any

from .logging import get_logger

logger = get_logger(__name__)

def timing_decorator(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"⏱️  {func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"⏱️  {func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper

class PerformanceMonitor:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"🚀 Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            logger.info(f"✅ {self.operation_name} completed in {duration:.2f}s")
        else:
            logger.error(f"❌ {self.operation_name} failed after {duration:.2f}s")
