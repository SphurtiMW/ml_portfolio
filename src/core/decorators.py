"""
Advanced Decorators Module

This module provides professional decorators for:
- Performance timing and monitoring
- Retry logic with exponential backoff
- Caching with TTL and LRU eviction
- Rate limiting and throttling
- Input validation and type checking
- Async/sync function conversion
- Method-level logging
- Error handling and circuit breakers
"""

import asyncio
import functools
import time
import inspect
from typing import Any, Callable, Dict, Optional, Union, TypeVar, Generic
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from .logger import get_logger, performance_logger
from .exceptions import handle_error, ForecastingError

F = TypeVar('F', bound=Callable[..., Any])
logger = get_logger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "LRU"  # Least Recently Used
    TTL = "TTL"  # Time To Live
    LFU = "LFU"  # Least Frequently Used


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def access(self):
        """Update access metadata"""
        self.accessed_at = datetime.now()
        self.access_count += 1


class Cache:
    """Thread-safe cache with multiple eviction strategies"""
    
    def __init__(self, max_size: int = 128, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self._cache: Dict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
    
    def _evict(self):
        """Evict entries based on strategy"""
        with self._lock:
            if len(self._cache) < self.max_size:
                return
            
            if self.strategy == CacheStrategy.LRU:
                # Remove least recently used
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            elif self.strategy == CacheStrategy.LFU:
                # Remove least frequently used
                min_access_key = min(self._cache.keys(), 
                                   key=lambda k: self._cache[k].access_count)
                del self._cache[min_access_key]
            
            elif self.strategy == CacheStrategy.TTL:
                # Remove expired entries first, then oldest
                expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
                if expired_keys:
                    for key in expired_keys:
                        del self._cache[key]
                elif self._cache:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return None
            
            entry.access()
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            
            self._evict()
            
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                ttl=ttl
            )
            self._cache[key] = entry
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            expired_count = sum(1 for entry in self._cache.values() if entry.is_expired())
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "strategy": self.strategy.value,
                "expired_entries": expired_count
            }


# Global cache instance
_global_cache = Cache(max_size=256)


def timer(func: F = None, *, logger_name: Optional[str] = None) -> F:
    """
    Decorator to measure function execution time
    
    Args:
        func: Function to decorate
        logger_name: Custom logger name
    
    Returns:
        Decorated function
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            func_logger = get_logger(logger_name or f.__module__)
            
            try:
                result = f(*args, **kwargs)
                duration = time.perf_counter() - start_time
                
                func_logger.info(
                    f"Function '{f.__name__}' completed",
                    function=f.__name__,
                    duration_seconds=duration,
                    status="success"
                )
                
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                func_logger.error(
                    f"Function '{f.__name__}' failed",
                    function=f.__name__,
                    duration_seconds=duration,
                    status="error",
                    error=str(e)
                )
                raise
        
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            func_logger = get_logger(logger_name or f.__module__)
            
            try:
                result = await f(*args, **kwargs)
                duration = time.perf_counter() - start_time
                
                func_logger.info(
                    f"Async function '{f.__name__}' completed",
                    function=f.__name__,
                    duration_seconds=duration,
                    status="success"
                )
                
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                func_logger.error(
                    f"Async function '{f.__name__}' failed",
                    function=f.__name__,
                    duration_seconds=duration,
                    status="error",
                    error=str(e)
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(f) else wrapper
    
    return decorator(func) if func else decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    logger_name: Optional[str] = None
):
    """
    Decorator for retry logic with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch for retry
        logger_name: Custom logger name
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = get_logger(logger_name or func.__module__)
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        func_logger.error(
                            f"Function '{func.__name__}' failed after {max_attempts} attempts",
                            function=func.__name__,
                            max_attempts=max_attempts,
                            final_error=str(e)
                        )
                        raise handle_error(e)
                    
                    func_logger.warning(
                        f"Function '{func.__name__}' failed on attempt {attempt + 1}, retrying",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        error=str(e),
                        retry_delay=current_delay
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_logger = get_logger(logger_name or func.__module__)
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        func_logger.error(
                            f"Async function '{func.__name__}' failed after {max_attempts} attempts",
                            function=func.__name__,
                            max_attempts=max_attempts,
                            final_error=str(e)
                        )
                        raise handle_error(e)
                    
                    func_logger.warning(
                        f"Async function '{func.__name__}' failed on attempt {attempt + 1}, retrying",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        error=str(e),
                        retry_delay=current_delay
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


def cache_result(
    ttl: Optional[float] = None,
    max_size: Optional[int] = None,
    strategy: CacheStrategy = CacheStrategy.LRU,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time to live in seconds
        max_size: Maximum cache size
        strategy: Cache eviction strategy
        key_func: Function to generate cache key
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        cache = Cache(max_size=max_size or 128, strategy=strategy)
        
        def make_key(*args, **kwargs) -> str:
            if key_func:
                return key_func(*args, **kwargs)
            
            # Generate key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            return "|".join(key_parts)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = make_key(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            
            return result
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = make_key(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(key, result, ttl)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.stats
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


class RateLimiter:
    """Thread-safe rate limiter"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self._calls = []
        self._lock = threading.RLock()
    
    def is_allowed(self) -> bool:
        """Check if call is allowed within rate limit"""
        with self._lock:
            now = time.time()
            
            # Remove old calls outside time window
            self._calls = [call_time for call_time in self._calls 
                          if now - call_time < self.time_window]
            
            if len(self._calls) < self.max_calls:
                self._calls.append(now)
                return True
            
            return False
    
    def wait_time(self) -> float:
        """Get time to wait before next allowed call"""
        with self._lock:
            if len(self._calls) < self.max_calls:
                return 0.0
            
            oldest_call = min(self._calls)
            return self.time_window - (time.time() - oldest_call)


def rate_limit(max_calls: int, time_window: float, wait: bool = True):
    """
    Decorator for rate limiting function calls
    
    Args:
        max_calls: Maximum number of calls
        time_window: Time window in seconds
        wait: Whether to wait for rate limit reset or raise exception
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        limiter = RateLimiter(max_calls, time_window)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not limiter.is_allowed():
                if wait:
                    wait_time = limiter.wait_time()
                    time.sleep(wait_time)
                else:
                    from .exceptions import RateLimitError
                    raise RateLimitError(max_calls, int(time_window))
            
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not limiter.is_allowed():
                if wait:
                    wait_time = limiter.wait_time()
                    await asyncio.sleep(wait_time)
                else:
                    from .exceptions import RateLimitError
                    raise RateLimitError(max_calls, int(time_window))
            
            return await func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


def validate_input(**validators):
    """
    Decorator for input validation
    
    Args:
        validators: Dictionary of parameter validators
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind arguments to signature
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate parameters
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        from .exceptions import DataValidationError
                        raise DataValidationError(
                            field=param_name,
                            value=value,
                            expected="valid input according to validator"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def log_method(include_args: bool = False, include_result: bool = False):
    """
    Decorator for logging method calls
    
    Args:
        include_args: Whether to log method arguments
        include_result: Whether to log method result
    
    Returns:
        Decorated method
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            class_name = self.__class__.__name__
            method_name = func.__name__
            func_logger = get_logger(f"{func.__module__}.{class_name}")
            
            log_data = {
                'class': class_name,
                'method': method_name
            }
            
            if include_args:
                log_data['args'] = str(args)[:200]
                log_data['kwargs'] = {k: str(v)[:100] for k, v in kwargs.items()}
            
            func_logger.info(f"Method called: {class_name}.{method_name}", **log_data)
            
            try:
                result = func(self, *args, **kwargs)
                
                if include_result:
                    log_data['result'] = str(result)[:200]
                
                func_logger.info(f"Method completed: {class_name}.{method_name}", **log_data)
                return result
                
            except Exception as e:
                func_logger.error(
                    f"Method failed: {class_name}.{method_name}",
                    class_name=class_name,
                    method=method_name,
                    error=str(e),
                    exc_info=True
                )
                raise
        
        return wrapper
    
    return decorator


def to_async(executor: Optional[ThreadPoolExecutor] = None):
    """
    Decorator to convert sync function to async
    
    Args:
        executor: Thread pool executor to use
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(executor, 
                                            functools.partial(func, *args, **kwargs))
        return wrapper
    
    return decorator


def to_sync(timeout: Optional[float] = None):
    """
    Decorator to convert async function to sync
    
    Args:
        timeout: Timeout for async operation
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if timeout:
                return loop.run_until_complete(
                    asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                )
            else:
                return loop.run_until_complete(func(*args, **kwargs))
        
        return wrapper
    
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (self.state == "OPEN" and 
                self.last_failure_time and
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise ForecastingError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """
    Decorator implementing circuit breaker pattern
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Timeout before attempting recovery
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        breaker = CircuitBreaker(failure_threshold, recovery_timeout)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


# Export decorators
__all__ = [
    'timer',
    'retry',
    'cache_result',
    'rate_limit',
    'validate_input',
    'log_method',
    'to_async',
    'to_sync',
    'circuit_breaker',
    'Cache',
    'CacheStrategy',
    'RateLimiter',
    'CircuitBreaker'
]