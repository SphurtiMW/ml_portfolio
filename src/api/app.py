"""
Professional FastAPI Application

This module creates and configures the main FastAPI application with:
- Async request handling
- Custom middleware stack
- Error handling and validation
- Authentication and security
- Monitoring and metrics
- API documentation
- CORS and rate limiting
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
import structlog

from ..core.config import settings
from ..core.logger import get_logger, setup_logging
from ..core.exceptions import BaseForecastingError, handle_error
from .middleware import (
    LoggingMiddleware,
    MetricsMiddleware, 
    RateLimitMiddleware,
    SecurityMiddleware
)
from .routes import api_router
from .auth import AuthManager

logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'http_active_connections',
    'Number of active HTTP connections'
)

MODEL_PREDICTIONS = Counter(
    'model_predictions_total',
    'Total number of model predictions',
    ['model_type', 'status']
)

MODEL_LATENCY = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction duration in seconds',
    ['model_type']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events
    """
    # Startup
    logger.info("Starting up application")
    
    # Initialize authentication
    app.state.auth_manager = AuthManager()
    
    # Initialize any background tasks
    app.state.background_tasks = set()
    
    # Start health check task
    health_task = asyncio.create_task(health_check_loop())
    app.state.background_tasks.add(health_task)
    
    logger.info("Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Cancel background tasks
    for task in app.state.background_tasks:
        task.cancel()
    
    # Wait for tasks to complete
    await asyncio.gather(*app.state.background_tasks, return_exceptions=True)
    
    logger.info("Application shutdown completed")


async def health_check_loop():
    """Background task for periodic health checks"""
    while True:
        try:
            # Perform health checks
            await asyncio.sleep(settings.monitoring.health_check_interval)
            
            # Add your health check logic here
            logger.debug("Health check performed")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health check failed: {e}")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    
    Returns:
        Configured FastAPI application instance
    """
    # Setup logging first
    setup_logging()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.api.title,
        description=settings.api.description,
        version=settings.api.version,
        debug=settings.api.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.api.debug else None,
        redoc_url="/redoc" if settings.api.debug else None,
        openapi_url="/openapi.json" if settings.api.debug else None
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Setup routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Setup error handlers
    setup_error_handlers(app)
    
    # Setup metrics endpoint if enabled
    if settings.monitoring.prometheus_enabled:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
    
    logger.info("FastAPI application created and configured")
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """
    Setup middleware stack
    
    Args:
        app: FastAPI application instance
    """
    # Security headers
    app.add_middleware(SecurityMiddleware)
    
    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=settings.api.rate_limit_requests,
        time_window=settings.api.rate_limit_window
    )
    
    # Metrics collection
    app.add_middleware(MetricsMiddleware)
    
    # Request logging
    app.add_middleware(LoggingMiddleware)
    
    # Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # CORS
    if settings.api.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.allowed_origins,
            allow_credentials=True,
            allow_methods=settings.api.allowed_methods,
            allow_headers=settings.api.allowed_headers,
        )
    
    # Trusted host middleware for production
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure properly for production
        )
    
    logger.info("Middleware stack configured")


def setup_error_handlers(app: FastAPI) -> None:
    """
    Setup custom error handlers
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(BaseForecastingError)
    async def forecasting_error_handler(request: Request, exc: BaseForecastingError):
        """Handle custom forecasting errors"""
        logger.error(
            "Forecasting error occurred",
            error_code=exc.error_code,
            error_category=exc.category.value,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.user_message,
                    "category": exc.category.value,
                    "timestamp": exc.context.timestamp.isoformat()
                }
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError exceptions"""
        structured_error = handle_error(exc)
        logger.error("ValueError occurred", error=str(exc), path=request.url.path)
        
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": structured_error.error_code,
                    "message": structured_error.user_message,
                    "category": structured_error.category.value
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        structured_error = handle_error(exc)
        logger.error(
            "Unhandled exception occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path,
            method=request.method,
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred. Please try again later.",
                    "category": "SYSTEM_ERROR"
                }
            }
        )
    
    logger.info("Error handlers configured")


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.api.title,
        version=settings.api.version,
        description=settings.api.description,
        routes=app.routes,
    )
    
    # Add custom security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Health check endpoints
async def health_check() -> Dict[str, Any]:
    """
    Application health check
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.api.version,
        "environment": settings.environment
    }


async def readiness_check() -> Dict[str, Any]:
    """
    Application readiness check
    
    Returns:
        Readiness status information
    """
    # Check database connectivity
    db_healthy = True  # Implement actual DB check
    
    # Check model availability
    model_healthy = True  # Implement actual model check
    
    ready = db_healthy and model_healthy
    
    return {
        "ready": ready,
        "checks": {
            "database": "healthy" if db_healthy else "unhealthy",
            "model": "healthy" if model_healthy else "unhealthy"
        },
        "timestamp": time.time()
    }


# Create the app instance
app = create_app()

# Override OpenAPI
app.openapi = custom_openapi


def run_server():
    """Run the development server"""
    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level=settings.logging.level.lower(),
        access_log=True,
        loop="asyncio"
    )


if __name__ == "__main__":
    run_server()