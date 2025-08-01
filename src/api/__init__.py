"""
API Module

This module provides professional REST API components:
- FastAPI application with async endpoints
- Authentication and authorization
- Request/response models with Pydantic
- Error handling and monitoring
- Rate limiting and caching
- OpenAPI documentation
"""

from .app import create_app
from .routes import router
from .middleware import setup_middleware
from .auth import AuthManager
from .models import *

__all__ = [
    "create_app",
    "router", 
    "setup_middleware",
    "AuthManager"
]