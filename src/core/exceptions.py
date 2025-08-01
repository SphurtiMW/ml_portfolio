"""
Advanced Exception Handling System

This module provides:
- Custom exception hierarchy with inheritance
- Structured error information
- Error codes and categorization
- Context preservation
- Integration with logging and monitoring
"""

import sys
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field


class ErrorCategory(Enum):
    """Error categories for classification"""
    DATA_ERROR = "DATA_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    API_ERROR = "API_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ErrorContext:
    """Structure to hold error context information"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


class BaseForecastingError(Exception):
    """Base exception class for all forecasting-related errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.context = context or self._get_error_context()
        self.cause = cause
        self.user_message = user_message or self._get_user_friendly_message()
    
    def _get_error_context(self) -> ErrorContext:
        """Extract context information from the current stack frame"""
        frame = sys._getframe(2)  # Skip current and __init__ frames
        return ErrorContext(
            file_path=frame.f_code.co_filename,
            line_number=frame.f_lineno,
            function_name=frame.f_code.co_name
        )
    
    def _get_user_friendly_message(self) -> str:
        """Generate user-friendly error message"""
        return "An error occurred while processing your request. Please try again later."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.context.timestamp.isoformat(),
            "context": {
                "file_path": self.context.file_path,
                "line_number": self.context.line_number,
                "function_name": self.context.function_name,
                "user_id": self.context.user_id,
                "request_id": self.context.request_id,
                "additional_data": self.context.additional_data
            },
            "cause": str(self.cause) if self.cause else None,
            "traceback": traceback.format_exc()
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(error_code='{self.error_code}', message='{self.message}')"


class ForecastingError(BaseForecastingError):
    """General forecasting error"""
    
    def __init__(self, message: str, cause: Optional[Exception] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="FORECASTING_001",
            category=ErrorCategory.SYSTEM_ERROR,
            cause=cause,
            **kwargs
        )


class DataError(BaseForecastingError):
    """Data-related errors"""
    pass


class DataNotFoundError(DataError):
    """Raised when required data is not found"""
    
    def __init__(self, resource: str, identifier: Optional[str] = None, **kwargs):
        message = f"Data not found: {resource}"
        if identifier:
            message += f" (ID: {identifier})"
        
        super().__init__(
            message=message,
            error_code="DATA_001",
            category=ErrorCategory.DATA_ERROR,
            user_message=f"The requested {resource} could not be found.",
            **kwargs
        )


class DataValidationError(DataError):
    """Raised when data validation fails"""
    
    def __init__(self, field: str, value: Any, expected: str, **kwargs):
        message = f"Invalid data for field '{field}': got {value}, expected {expected}"
        
        super().__init__(
            message=message,
            error_code="DATA_002",
            category=ErrorCategory.VALIDATION_ERROR,
            user_message=f"Invalid value provided for {field}. {expected}",
            **kwargs
        )


class DataFormatError(DataError):
    """Raised when data format is incorrect"""
    
    def __init__(self, expected_format: str, actual_format: str, **kwargs):
        message = f"Data format error: expected {expected_format}, got {actual_format}"
        
        super().__init__(
            message=message,
            error_code="DATA_003",
            category=ErrorCategory.DATA_ERROR,
            user_message="The data format is not supported. Please check your input.",
            **kwargs
        )


class ModelError(BaseForecastingError):
    """Model-related errors"""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a model is not found"""
    
    def __init__(self, model_name: str, version: Optional[str] = None, **kwargs):
        message = f"Model '{model_name}' not found"
        if version:
            message += f" (version: {version})"
        
        super().__init__(
            message=message,
            error_code="MODEL_001",
            category=ErrorCategory.MODEL_ERROR,
            user_message="The requested model is not available.",
            **kwargs
        )


class ModelTrainingError(ModelError):
    """Raised when model training fails"""
    
    def __init__(self, stage: str, details: Optional[str] = None, **kwargs):
        message = f"Model training failed at stage: {stage}"
        if details:
            message += f" - {details}"
        
        super().__init__(
            message=message,
            error_code="MODEL_002",
            category=ErrorCategory.MODEL_ERROR,
            severity=ErrorSeverity.HIGH,
            user_message="Model training encountered an error. Please check your data and try again.",
            **kwargs
        )


class ModelPredictionError(ModelError):
    """Raised when model prediction fails"""
    
    def __init__(self, model_name: str, input_shape: Optional[tuple] = None, **kwargs):
        message = f"Model prediction failed for '{model_name}'"
        if input_shape:
            message += f" with input shape {input_shape}"
        
        super().__init__(
            message=message,
            error_code="MODEL_003",
            category=ErrorCategory.MODEL_ERROR,
            user_message="Unable to generate predictions. Please check your input data.",
            **kwargs
        )


class ModelVersionError(ModelError):
    """Raised when there are model versioning issues"""
    
    def __init__(self, operation: str, version: str, **kwargs):
        message = f"Model version error during {operation}: version {version}"
        
        super().__init__(
            message=message,
            error_code="MODEL_004",
            category=ErrorCategory.MODEL_ERROR,
            user_message="There was an issue with the model version. Please try again.",
            **kwargs
        )


class APIError(BaseForecastingError):
    """API-related errors"""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails"""
    
    def __init__(self, details: Optional[str] = None, **kwargs):
        message = "Authentication failed"
        if details:
            message += f": {details}"
        
        super().__init__(
            message=message,
            error_code="AUTH_001",
            category=ErrorCategory.AUTHENTICATION_ERROR,
            user_message="Authentication failed. Please check your credentials.",
            **kwargs
        )


class AuthorizationError(APIError):
    """Raised when authorization fails"""
    
    def __init__(self, resource: str, action: str, **kwargs):
        message = f"Access denied for {action} on {resource}"
        
        super().__init__(
            message=message,
            error_code="AUTH_002",
            category=ErrorCategory.AUTHORIZATION_ERROR,
            user_message="You don't have permission to perform this action.",
            **kwargs
        )


class RateLimitError(APIError):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, limit: int, window: int, **kwargs):
        message = f"Rate limit exceeded: {limit} requests per {window} seconds"
        
        super().__init__(
            message=message,
            error_code="API_001",
            category=ErrorCategory.API_ERROR,
            user_message=f"Too many requests. Please wait {window} seconds before trying again.",
            **kwargs
        )


class ExternalServiceError(BaseForecastingError):
    """Raised when external service calls fail"""
    
    def __init__(self, service: str, operation: str, status_code: Optional[int] = None, **kwargs):
        message = f"External service error: {service} - {operation}"
        if status_code:
            message += f" (status: {status_code})"
        
        super().__init__(
            message=message,
            error_code="EXT_001",
            category=ErrorCategory.EXTERNAL_SERVICE_ERROR,
            user_message="An external service is currently unavailable. Please try again later.",
            **kwargs
        )


class ConfigurationError(BaseForecastingError):
    """Raised when configuration is invalid"""
    
    def __init__(self, parameter: str, value: Any, expected: str, **kwargs):
        message = f"Invalid configuration for '{parameter}': got {value}, expected {expected}"
        
        super().__init__(
            message=message,
            error_code="CONFIG_001",
            category=ErrorCategory.SYSTEM_ERROR,
            severity=ErrorSeverity.CRITICAL,
            user_message="System configuration error. Please contact support.",
            **kwargs
        )


# Utility functions for error handling
def error_message_detail(error: Exception, error_detail: Any = None) -> str:
    """
    Extract detailed error information
    
    Args:
        error: The exception instance
        error_detail: Additional error details (sys module for traceback)
    
    Returns:
        Formatted error message with details
    """
    if error_detail and hasattr(error_detail, 'exc_info'):
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb:
            filename = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error in {filename}:{line_number} - {str(error)}"
    
    return str(error)


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> BaseForecastingError:
    """
    Convert generic exceptions to structured forecasting errors
    
    Args:
        error: The original exception
        context: Additional context information
        user_id: User identifier
        request_id: Request identifier
    
    Returns:
        Structured forecasting error
    """
    error_context = ErrorContext(
        user_id=user_id,
        request_id=request_id,
        additional_data=context or {}
    )
    
    if isinstance(error, BaseForecastingError):
        return error
    
    # Map common exceptions to specific error types
    if isinstance(error, (FileNotFoundError, KeyError)):
        return DataNotFoundError(
            resource=str(error),
            context=error_context,
            cause=error
        )
    elif isinstance(error, (ValueError, TypeError)):
        return DataValidationError(
            field="unknown",
            value=str(error),
            expected="valid input",
            context=error_context,
            cause=error
        )
    elif isinstance(error, PermissionError):
        return AuthorizationError(
            resource="unknown",
            action="access",
            context=error_context,
            cause=error
        )
    else:
        return ForecastingError(
            message=f"Unexpected error: {str(error)}",
            context=error_context,
            cause=error
        )


# Custom exception for backwards compatibility
class CustomException(BaseForecastingError):
    """Legacy custom exception for backwards compatibility"""
    
    def __init__(self, error_message: str, error_detail: Any = None):
        context = None
        if error_detail and hasattr(error_detail, 'exc_info'):
            _, _, exc_tb = error_detail.exc_info()
            if exc_tb:
                context = ErrorContext(
                    file_path=exc_tb.tb_frame.f_code.co_filename,
                    line_number=exc_tb.tb_lineno,
                    function_name=exc_tb.tb_frame.f_code.co_name
                )
        
        super().__init__(
            message=error_message_detail(Exception(error_message), error_detail),
            error_code="LEGACY_001",
            category=ErrorCategory.SYSTEM_ERROR,
            context=context
        )