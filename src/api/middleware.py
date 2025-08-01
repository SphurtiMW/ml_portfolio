"""
Professional Middleware Stack

This module provides custom middleware for:
- Request/response logging with correlation IDs
- Metrics collection for Prometheus
- Rate limiting and throttling
- Security headers
- Request timing and performance monitoring
"""

import time
import uuid
import asyncio
from typing import Callable, Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta
import json

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from prometheus_client import Counter, Histogram, Gauge

from ..core.logger import get_logger
from ..core.config import settings

logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
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

REQUEST_SIZE = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

RESPONSE_SIZE = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger(f"{__name__}.LoggingMiddleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Start timing
        start_time = time.perf_counter()
        
        # Get request size
        request_size = int(request.headers.get("content-length", 0))
        
        # Log request
        self.logger.info(
            "HTTP request started",
            method=request.method,
            url=str(request.url),
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            correlation_id=correlation_id,
            request_size=request_size
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.perf_counter() - start_time
            
            # Get response size
            response_size = len(response.body) if hasattr(response, 'body') else 0
            
            # Log response
            self.logger.info(
                "HTTP request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_seconds=duration,
                correlation_id=correlation_id,
                response_size=response_size
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            
            self.logger.error(
                "HTTP request failed",
                method=request.method,
                path=request.url.path,
                duration_seconds=duration,
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An internal error occurred",
                        "correlation_id": correlation_id
                    }
                },
                headers={"X-Correlation-ID": correlation_id}
            )


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting Prometheus metrics
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger(f"{__name__}.MetricsMiddleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Track active connections
        ACTIVE_CONNECTIONS.inc()
        
        # Start timing
        start_time = time.perf_counter()
        
        # Get request info
        method = request.method
        path = request.url.path
        
        # Clean path for metrics (remove dynamic parts)
        endpoint = self._clean_path_for_metrics(path)
        
        # Get request size
        request_size = int(request.headers.get("content-length", 0))
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.perf_counter() - start_time
            
            # Get response size
            response_size = len(response.body) if hasattr(response, 'body') else 0
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            REQUEST_SIZE.labels(
                method=method,
                endpoint=endpoint
            ).observe(request_size)
            
            RESPONSE_SIZE.labels(
                method=method,
                endpoint=endpoint
            ).observe(response_size)
            
            return response
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            
            # Record error metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=500
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            raise
            
        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()
    
    def _clean_path_for_metrics(self, path: str) -> str:
        """Clean path for metrics to avoid high cardinality"""
        # Replace UUIDs and IDs with placeholders
        import re
        
        # Replace UUID patterns
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Limit path length
        if len(path) > 100:
            path = path[:97] + "..."
        
        return path


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests
    """
    
    def __init__(self, app: ASGIApp, max_requests: int = 100, time_window: int = 3600):
        super().__init__(app)
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, list] = defaultdict(list)
        self.logger = get_logger(f"{__name__}.RateLimitMiddleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_ip = self._get_client_identifier(request)
        
        # Check rate limit
        if not self._is_allowed(client_ip):
            self.logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                max_requests=self.max_requests,
                time_window=self.time_window
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Maximum {self.max_requests} requests per {self.time_window} seconds.",
                        "retry_after": self._get_retry_after(client_ip)
                    }
                },
                headers={
                    "Retry-After": str(self._get_retry_after(client_ip))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.time_window)
        
        return response
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get real IP from headers (if behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to client IP
        return request.client.host if request.client else "unknown"
    
    def _is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.time_window
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
        
        return False
    
    def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        now = time.time()
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.time_window
        ]
        
        return max(0, self.max_requests - len(self.requests[client_id]))
    
    def _get_retry_after(self, client_id: str) -> int:
        """Get retry after time in seconds"""
        if not self.requests[client_id]:
            return 0
        
        oldest_request = min(self.requests[client_id])
        return max(0, int(self.time_window - (time.time() - oldest_request)))


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger(f"{__name__}.SecurityMiddleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """
    Middleware for limiting request size
    """
    
    def __init__(self, app: ASGIApp, max_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_size = max_size
        self.logger = get_logger(f"{__name__}.RequestSizeMiddleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = int(request.headers.get("content-length", 0))
        
        if content_length > self.max_size:
            self.logger.warning(
                "Request size exceeded",
                content_length=content_length,
                max_size=self.max_size,
                client_ip=request.client.host if request.client else None
            )
            
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "REQUEST_TOO_LARGE",
                        "message": f"Request size {content_length} exceeds maximum allowed size {self.max_size}",
                        "max_size": self.max_size
                    }
                }
            )
        
        return await call_next(request)


class ResponseTimeMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding response time headers
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger(f"{__name__}.ResponseTimeMiddleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


# Utility function to setup all middleware
def setup_middleware(app):
    """
    Setup all middleware on the FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    from ..core.config import settings
    
    # Add middleware in reverse order (last added is executed first)
    
    # Response time (should be outermost)
    app.add_middleware(ResponseTimeMiddleware)
    
    # Security headers
    app.add_middleware(SecurityMiddleware)
    
    # Request size limiting
    app.add_middleware(RequestSizeMiddleware, max_size=10 * 1024 * 1024)  # 10MB
    
    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=settings.api.rate_limit_requests,
        time_window=settings.api.rate_limit_window
    )
    
    # Metrics collection
    app.add_middleware(MetricsMiddleware)
    
    # Request/response logging (should be innermost)
    app.add_middleware(LoggingMiddleware)
    
    logger.info("All middleware configured successfully")


# Export middleware classes
__all__ = [
    "LoggingMiddleware",
    "MetricsMiddleware",
    "RateLimitMiddleware",
    "SecurityMiddleware",
    "RequestSizeMiddleware",
    "ResponseTimeMiddleware",
    "setup_middleware"
]