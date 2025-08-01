"""
Time Series Based Supply Chain Demand Forecasting Using LSTM
Professional ML Portfolio Project with Advanced Python Concepts

This package provides a comprehensive solution for demand forecasting
using LSTM networks with MLOps practices, async programming, and
professional software development patterns.
"""

__version__ = "2.0.0"
__author__ = "SphurtiMW"
__email__ = "sphurtimw@gmail.com"

from src.core.config import settings
from src.core.logger import get_logger
from src.core.exceptions import ForecastingError

# Package-level logger
logger = get_logger(__name__)

__all__ = [
    "settings",
    "get_logger", 
    "ForecastingError",
    "__version__",
    "__author__",
    "__email__"
]