"""
API Models Module

This module defines Pydantic models for request/response validation:
- Request models with validation rules
- Response models with proper serialization
- Error models for consistent error handling
- Complex nested models for forecasting
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class DataPoint(BaseModel):
    """Single data point for time series"""
    date: Optional[datetime] = Field(None, description="Date of the observation")
    demand: float = Field(..., gt=0, description="Demand value (must be positive)")
    
    class Config:
        schema_extra = {
            "example": {
                "date": "2023-01-01T00:00:00",
                "demand": 1250.5
            }
        }


class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions"""
    lower: float = Field(..., description="Lower bound of confidence interval")
    upper: float = Field(..., description="Upper bound of confidence interval")
    
    @validator('upper')
    def upper_must_be_greater_than_lower(cls, v, values):
        if 'lower' in values and v <= values['lower']:
            raise ValueError('Upper bound must be greater than lower bound')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "lower": 1150.0,
                "upper": 1350.0
            }
        }


class PredictionRequest(BaseModel):
    """Request model for demand prediction"""
    historical_data: List[DataPoint] = Field(
        ..., 
        min_items=30,
        max_items=10000,
        description="Historical demand data (minimum 30 points required)"
    )
    forecast_horizon: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of periods to forecast (1-365)"
    )
    sequence_length: int = Field(
        default=60,
        ge=10,
        le=200,
        description="LSTM sequence length (10-200)"
    )
    include_confidence_intervals: bool = Field(
        default=True,
        description="Whether to include confidence intervals in response"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.8,
        le=0.99,
        description="Confidence level for intervals (0.8-0.99)"
    )
    
    @validator('historical_data')
    def validate_historical_data_sequence(cls, v):
        if len(v) < 30:
            raise ValueError('At least 30 historical data points are required')
        
        # Check for reasonable demand values
        demands = [point.demand for point in v]
        if max(demands) / min(demands) > 1000:
            raise ValueError('Demand values have unrealistic variance')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "historical_data": [
                    {"date": "2023-01-01T00:00:00", "demand": 1200.0},
                    {"date": "2023-01-02T00:00:00", "demand": 1150.5},
                    {"date": "2023-01-03T00:00:00", "demand": 1300.2}
                ],
                "forecast_horizon": 30,
                "sequence_length": 60,
                "include_confidence_intervals": True,
                "confidence_level": 0.95
            }
        }


class PredictionResponse(BaseModel):
    """Response model for demand prediction"""
    predictions: List[float] = Field(..., description="Forecasted demand values")
    confidence_intervals: Optional[List[ConfidenceInterval]] = Field(
        None, 
        description="Confidence intervals for each prediction"
    )
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")
    
    @validator('predictions')
    def validate_predictions(cls, v):
        if not v:
            raise ValueError('Predictions list cannot be empty')
        if any(p <= 0 for p in v):
            raise ValueError('All predictions must be positive')
        return v
    
    @root_validator
    def validate_confidence_intervals_length(cls, values):
        predictions = values.get('predictions', [])
        confidence_intervals = values.get('confidence_intervals')
        
        if confidence_intervals is not None:
            if len(confidence_intervals) != len(predictions):
                raise ValueError('Confidence intervals must match predictions length')
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [1250.5, 1275.3, 1290.1],
                "confidence_intervals": [
                    {"lower": 1150.0, "upper": 1350.0},
                    {"lower": 1175.0, "upper": 1375.0},
                    {"lower": 1190.0, "upper": 1390.0}
                ],
                "metadata": {
                    "model_version": "2.0.0",
                    "sequence_length": 60,
                    "forecast_horizon": 3,
                    "processing_time": 0.245,
                    "confidence_level": 0.95
                }
            }
        }


class ModelStatus(str, Enum):
    """Model status enumeration"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    TRAINING = "training"
    FAILED = "failed"


class ModelType(str, Enum):
    """Model type enumeration"""
    LSTM = "LSTM"
    ARIMA = "ARIMA"
    PROPHET = "PROPHET"
    LINEAR = "LINEAR"


class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: ModelType = Field(..., description="Model type")
    status: ModelStatus = Field(..., description="Model status")
    created_at: datetime = Field(..., description="Model creation timestamp")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    description: Optional[str] = Field(None, description="Model description")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "lstm_demand_forecaster",
                "version": "2.0.0",
                "type": "LSTM",
                "status": "active",
                "created_at": "2023-01-01T00:00:00",
                "metrics": {
                    "mae": 145.2,
                    "rmse": 203.7,
                    "mape": 8.5
                },
                "description": "Advanced LSTM model for demand forecasting"
            }
        }


class TrainingConfig(BaseModel):
    """Model training configuration"""
    epochs: int = Field(default=100, ge=10, le=1000, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=8, le=512, description="Training batch size")
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1, description="Learning rate")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.3, description="Validation data split")
    early_stopping_patience: int = Field(default=10, ge=5, le=50, description="Early stopping patience")
    
    class Config:
        schema_extra = {
            "example": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "early_stopping_patience": 10
            }
        }


class TrainingRequest(BaseModel):
    """Request model for model training"""
    training_data: List[DataPoint] = Field(
        ..., 
        min_items=100,
        description="Training data (minimum 100 points required)"
    )
    config: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    model_name: Optional[str] = Field(None, description="Custom model name")
    description: Optional[str] = Field(None, description="Training job description")
    
    @validator('training_data')
    def validate_training_data(cls, v):
        if len(v) < 100:
            raise ValueError('At least 100 training data points are required')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "training_data": [
                    {"date": "2023-01-01T00:00:00", "demand": 1200.0},
                    {"date": "2023-01-02T00:00:00", "demand": 1150.5}
                ],
                "config": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001
                },
                "model_name": "custom_lstm_v1",
                "description": "Training with new seasonal data"
            }
        }


class TrainingStatus(str, Enum):
    """Training job status"""
    QUEUED = "queued"
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingResponse(BaseModel):
    """Response model for training request"""
    job_id: str = Field(..., description="Training job identifier")
    status: TrainingStatus = Field(..., description="Current training status")
    message: str = Field(..., description="Status message")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Training progress percentage")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "train_20231201_143022",
                "status": "started",
                "message": "Model training job started successfully",
                "estimated_duration": 3600,
                "progress": 0.0
            }
        }


class HealthStatus(str, Enum):
    """Health check status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthResponse(BaseModel):
    """Health check response"""
    status: HealthStatus = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment name")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2023-01-01T12:00:00",
                "version": "2.0.0",
                "environment": "production",
                "uptime": 86400.0
            }
        }


class ErrorDetail(BaseModel):
    """Error detail information"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    
    class Config:
        schema_extra = {
            "example": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid input data format",
                "field": "historical_data"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: ErrorDetail = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "DATA_VALIDATION_ERROR",
                    "message": "Historical data must contain at least 30 points",
                    "field": "historical_data"
                },
                "timestamp": "2023-01-01T12:00:00",
                "request_id": "req_abc123"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    requests: List[PredictionRequest] = Field(
        ..., 
        min_items=1,
        max_items=100,
        description="List of prediction requests (max 100)"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Whether to process requests in parallel"
    )
    
    @validator('requests')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError('Batch size cannot exceed 100 requests')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "requests": [
                    {
                        "historical_data": [
                            {"date": "2023-01-01T00:00:00", "demand": 1200.0}
                        ],
                        "forecast_horizon": 30
                    }
                ],
                "parallel_processing": True
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    results: List[Union[PredictionResponse, ErrorResponse]] = Field(
        ..., 
        description="List of prediction results or errors"
    )
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "predictions": [1250.5, 1275.3],
                        "confidence_intervals": [],
                        "metadata": {"model_version": "2.0.0"}
                    }
                ],
                "summary": {
                    "total_requests": 1,
                    "successful": 1,
                    "failed": 0,
                    "processing_time": 1.23
                }
            }
        }


class UserInfo(BaseModel):
    """User information model"""
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="User email")
    roles: List[str] = Field(default=[], description="User roles")
    permissions: List[str] = Field(default=[], description="User permissions")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "username": "john_doe",
                "email": "john@example.com",
                "roles": ["analyst", "forecaster"],
                "permissions": ["predict", "upload_data"]
            }
        }


class TokenResponse(BaseModel):
    """Authentication token response"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600
            }
        }


# Export all models
__all__ = [
    "DataPoint",
    "ConfidenceInterval", 
    "PredictionRequest",
    "PredictionResponse",
    "ModelInfo",
    "ModelStatus",
    "ModelType",
    "TrainingConfig",
    "TrainingRequest",
    "TrainingResponse",
    "TrainingStatus",
    "HealthResponse",
    "HealthStatus",
    "ErrorDetail",
    "ErrorResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "UserInfo",
    "TokenResponse"
]