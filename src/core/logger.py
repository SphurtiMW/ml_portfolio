"""
Advanced Logging System with Structured Logging

This module provides:
- Structured logging with JSON format
- Correlation ID tracking
- Performance monitoring
- Context-aware logging
- Log aggregation support
- Async logging capabilities
"""

import os
import sys
import json
import asyncio
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone
from contextlib import contextmanager
from functools import wraps
import structlog
from structlog.typing import FilteringBoundLogger

from .config import settings


class CorrelationIDFilter(logging.Filter):
    """Filter to add correlation ID to log records"""
    
    def filter(self, record):
        record.correlation_id = getattr(record, 'correlation_id', 'N/A')
        record.request_id = getattr(record, 'request_id', 'N/A')
        record.user_id = getattr(record, 'user_id', 'N/A')
        return True


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logs"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
            'correlation_id': getattr(record, 'correlation_id', None),
            'request_id': getattr(record, 'request_id', None),
            'user_id': getattr(record, 'user_id', None)
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info', 'correlation_id', 'request_id', 'user_id']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class AsyncFileHandler(logging.Handler):
    """Async file handler for non-blocking logging"""
    
    def __init__(self, filename: str, mode: str = 'a'):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self._queue = asyncio.Queue()
        self._task = None
    
    async def _writer(self):
        """Async writer coroutine"""
        while True:
            try:
                record = await self._queue.get()
                if record is None:  # Shutdown signal
                    break
                
                formatted = self.format(record)
                async with aiofiles.open(self.filename, self.mode) as f:
                    await f.write(formatted + '\n')
                    await f.flush()
                
                self._queue.task_done()
            except Exception as e:
                print(f"Error in async logging: {e}", file=sys.stderr)
    
    def emit(self, record):
        """Emit a log record"""
        if self._task is None:
            loop = asyncio.get_event_loop()
            self._task = loop.create_task(self._writer())
        
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                self._queue.put_nowait, record
            )
        except Exception:
            self.handleError(record)
    
    def close(self):
        """Close the handler"""
        if self._task:
            asyncio.get_event_loop().call_soon_threadsafe(
                self._queue.put_nowait, None
            )
        super().close()


def setup_structlog():
    """Configure structlog for structured logging"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.add_logger_name,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer() if settings.logging.json_format else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(settings.logging.level),
        logger_factory=structlog.WriteLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=False,
    )


def setup_logging():
    """Setup comprehensive logging configuration"""
    # Create logs directory
    log_dir = settings.logging.file_path
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.logging.level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.logging.level.upper()))
    
    if settings.logging.json_format:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(settings.logging.format)
    
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(CorrelationIDFilter())
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if settings.logging.file_enabled:
        log_file = log_dir / f"{settings.project_name}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=settings.logging.max_file_size,
            backupCount=settings.logging.backup_count,
            encoding='utf-8'
        )
        
        file_handler.setLevel(getattr(logging, settings.logging.level.upper()))
        file_handler.setFormatter(JSONFormatter() if settings.logging.json_format else logging.Formatter(settings.logging.format))
        file_handler.addFilter(CorrelationIDFilter())
        root_logger.addHandler(file_handler)
    
    # Error file handler
    error_log_file = log_dir / f"{settings.project_name}_errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        filename=error_log_file,
        maxBytes=settings.logging.max_file_size,
        backupCount=settings.logging.backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    error_handler.addFilter(CorrelationIDFilter())
    root_logger.addHandler(error_handler)
    
    # Setup structlog
    if settings.logging.structured:
        setup_structlog()
    
    # Log startup
    logger = get_logger(__name__)
    logger.info("Logging system initialized", 
                level=settings.logging.level,
                structured=settings.logging.structured,
                json_format=settings.logging.json_format)


def get_logger(name: str) -> Union[logging.Logger, FilteringBoundLogger]:
    """
    Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance (structlog or standard logging)
    """
    if settings.logging.structured:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding context information"""
    
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        # Add context from kwargs
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.performance")
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = datetime.now()
    
    def end_timer(self, operation: str, **context):
        """End timing and log the duration"""
        if operation in self.metrics:
            start_time = self.metrics.pop(operation)
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(
                f"Operation completed: {operation}",
                operation=operation,
                duration_seconds=duration,
                **context
            )
            
            return duration
        return None
    
    @contextmanager
    def timer(self, operation: str, **context):
        """Context manager for timing operations"""
        self.start_timer(operation)
        try:
            yield
        finally:
            self.end_timer(operation, **context)


class AuditLogger:
    """Logger for audit events"""
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    def log_user_action(self, user_id: str, action: str, resource: str, 
                       result: str, **context):
        """Log user actions for audit purposes"""
        self.logger.info(
            f"User action: {action}",
            event_type="user_action",
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            **context
        )
    
    def log_model_operation(self, operation: str, model_name: str,
                           version: Optional[str] = None, **context):
        """Log model operations"""
        self.logger.info(
            f"Model operation: {operation}",
            event_type="model_operation",
            operation=operation,
            model_name=model_name,
            model_version=version,
            **context
        )
    
    def log_data_access(self, user_id: str, dataset: str, operation: str, **context):
        """Log data access events"""
        self.logger.info(
            f"Data access: {operation}",
            event_type="data_access",
            user_id=user_id,
            dataset=dataset,
            operation=operation,
            **context
        )


# Context managers for correlation tracking
@contextmanager
def log_context(**context):
    """Context manager for adding context to logs"""
    if settings.logging.structured:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**context)
        try:
            yield
        finally:
            structlog.contextvars.clear_contextvars()
    else:
        # For standard logging, we'll use thread-local storage or similar
        yield


def log_function_call(include_args: bool = False, include_result: bool = False):
    """Decorator to log function calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            log_data = {
                'function': func.__name__,
                'module': func.__module__
            }
            
            if include_args:
                log_data['args'] = str(args)[:200]  # Truncate long args
                log_data['kwargs'] = {k: str(v)[:100] for k, v in kwargs.items()}
            
            logger.info(f"Function called: {func.__name__}", **log_data)
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                
                result_log = {
                    'function': func.__name__,
                    'duration_seconds': duration,
                    'status': 'success'
                }
                
                if include_result:
                    result_log['result'] = str(result)[:200]
                
                logger.info(f"Function completed: {func.__name__}", **result_log)
                return result
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Function failed: {func.__name__}",
                    function=func.__name__,
                    duration_seconds=duration,
                    status='error',
                    error=str(e),
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# Initialize logging on module import
if not logging.getLogger().handlers:
    setup_logging()


# Global logger instances
performance_logger = PerformanceLogger(__name__)
audit_logger = AuditLogger()


# Export main functions
__all__ = [
    'setup_logging',
    'get_logger',
    'LoggerAdapter',
    'PerformanceLogger',
    'AuditLogger',
    'log_context',
    'log_function_call',
    'performance_logger',
    'audit_logger'
]