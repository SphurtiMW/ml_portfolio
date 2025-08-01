"""
Core module containing fundamental components:
- Configuration management
- Custom exceptions
- Logging utilities
- Base classes and interfaces
- Design patterns implementations
"""

from .config import settings
from .exceptions import ForecastingError, ModelError, DataError
from .logger import get_logger, setup_logging
from .decorators import timer, retry, cache_result, log_method
from .context_managers import ModelManager, DatabaseConnection

__all__ = [
    "settings",
    "ForecastingError",
    "ModelError", 
    "DataError",
    "get_logger",
    "setup_logging",
    "timer",
    "retry",
    "cache_result",
    "log_method",
    "ModelManager",
    "DatabaseConnection"
]