"""
API Routes Module

This module defines all API endpoints with:
- Async endpoint handlers
- Request/response validation with Pydantic
- Authentication and authorization
- Error handling and monitoring
- OpenAPI documentation
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import io

from ..core.logger import get_logger
from ..core.decorators import timer, rate_limit
from ..core.context_managers import performance_monitor, async_performance_monitor
from ..data.processors import DataProcessorFactory, LSTMDataProcessor, LoggingObserver
from .models import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse
)
from .auth import AuthManager, get_current_user

logger = get_logger(__name__)
security = HTTPBearer()

# Create router
api_router = APIRouter()


@api_router.get("/health", 
                response_model=HealthResponse,
                tags=["Health"],
                summary="Health Check",
                description="Check the health status of the API")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status information
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="2.0.0",
        environment="development"
    )


@api_router.get("/readiness",
                response_model=Dict[str, Any],
                tags=["Health"],
                summary="Readiness Check",
                description="Check if the API is ready to serve requests")
async def readiness_check():
    """
    Readiness check endpoint
    
    Returns:
        Readiness status information
    """
    async with async_performance_monitor("readiness_check") as perf:
        # Simulate checks
        await asyncio.sleep(0.01)  # Database check simulation
        
        return {
            "ready": True,
            "checks": {
                "database": "healthy",
                "model": "healthy",
                "cache": "healthy"
            },
            "timestamp": datetime.utcnow(),
            "performance": {
                "check_duration": perf.get("duration_seconds", 0)
            }
        }


@api_router.post("/predict",
                 response_model=PredictionResponse,
                 tags=["Prediction"],
                 summary="Make Prediction",
                 description="Generate demand forecast predictions using LSTM model")
@timer
@rate_limit(max_calls=10, time_window=60)  # 10 requests per minute
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate demand forecast predictions
    
    Args:
        request: Prediction request with historical data
        background_tasks: FastAPI background tasks
        current_user: Authenticated user information
    
    Returns:
        Prediction response with forecasted values
    """
    async with async_performance_monitor("prediction", user_id=current_user.get("user_id")) as perf:
        try:
            logger.info("Prediction request received",
                       user_id=current_user.get("user_id"),
                       data_points=len(request.historical_data))
            
            # Convert request data to DataFrame
            df = pd.DataFrame(request.historical_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Create LSTM processor
            processor = DataProcessorFactory.create_processor(
                'lstm',
                sequence_length=request.sequence_length,
                target_column='demand'
            )
            
            # Add progress observer
            observer = LoggingObserver()
            processor.add_observer(observer)
            
            # Process data
            result = processor.process(df)
            
            if not result.success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Data processing failed: {'; '.join(result.errors)}"
                )
            
            # Generate predictions (simplified for demo)
            predictions = await generate_predictions_async(
                result.data,
                request.forecast_horizon
            )
            
            # Log prediction for audit
            background_tasks.add_task(
                log_prediction_audit,
                user_id=current_user.get("user_id"),
                request=request,
                predictions=predictions
            )
            
            response = PredictionResponse(
                predictions=predictions,
                confidence_intervals=[
                    {"lower": p * 0.9, "upper": p * 1.1} 
                    for p in predictions
                ],
                metadata={
                    "model_version": "2.0.0",
                    "sequence_length": request.sequence_length,
                    "forecast_horizon": request.forecast_horizon,
                    "processing_time": perf.get("duration_seconds", 0),
                    "data_points_processed": len(request.historical_data)
                }
            )
            
            logger.info("Prediction completed successfully",
                       user_id=current_user.get("user_id"),
                       predictions_count=len(predictions),
                       processing_time=perf.get("duration_seconds", 0))
            
            return response
            
        except Exception as e:
            logger.error("Prediction failed",
                        user_id=current_user.get("user_id"),
                        error=str(e),
                        exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )


@api_router.post("/predict/batch",
                 response_model=List[PredictionResponse],
                 tags=["Prediction"],
                 summary="Batch Prediction",
                 description="Generate predictions for multiple datasets")
async def batch_predict(
    requests: List[PredictionRequest],
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate batch predictions
    
    Args:
        requests: List of prediction requests
        background_tasks: FastAPI background tasks
        current_user: Authenticated user information
    
    Returns:
        List of prediction responses
    """
    async with async_performance_monitor("batch_prediction", user_id=current_user.get("user_id")) as perf:
        logger.info("Batch prediction request received",
                   user_id=current_user.get("user_id"),
                   batch_size=len(requests))
        
        # Process requests concurrently
        tasks = [
            process_single_prediction(req, current_user)
            for req in requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results and log errors
        responses = []
        error_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_count += 1
                logger.error(f"Batch prediction item {i} failed: {result}")
                # You might want to include error information in response
            else:
                responses.append(result)
        
        logger.info("Batch prediction completed",
                   user_id=current_user.get("user_id"),
                   total_requests=len(requests),
                   successful=len(responses),
                   errors=error_count,
                   processing_time=perf.get("duration_seconds", 0))
        
        return responses


@api_router.post("/train",
                 response_model=TrainingResponse,
                 tags=["Model"],
                 summary="Train Model",
                 description="Trigger model training with new data")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Train a new model with provided data
    
    Args:
        request: Training request with configuration
        background_tasks: FastAPI background tasks
        current_user: Authenticated user information
    
    Returns:
        Training response with job information
    """
    logger.info("Model training request received",
               user_id=current_user.get("user_id"),
               model_config=request.config)
    
    # Start training as background task
    job_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    background_tasks.add_task(
        start_training_job,
        job_id=job_id,
        request=request,
        user_id=current_user.get("user_id")
    )
    
    return TrainingResponse(
        job_id=job_id,
        status="started",
        message="Model training job started successfully",
        estimated_duration=request.config.get("epochs", 100) * 10  # Rough estimate
    )


@api_router.get("/models",
                response_model=List[ModelInfo],
                tags=["Model"],
                summary="List Models",
                description="Get list of available models")
async def list_models(
    current_user: Dict = Depends(get_current_user)
):
    """
    List available models
    
    Returns:
        List of model information
    """
    # This would typically query a model registry
    models = [
        ModelInfo(
            name="lstm_demand_forecaster",
            version="2.0.0",
            type="LSTM",
            status="active",
            created_at=datetime.utcnow() - timedelta(days=1),
            metrics={
                "mae": 145.2,
                "rmse": 203.7,
                "mape": 8.5
            }
        ),
        ModelInfo(
            name="lstm_demand_forecaster",
            version="1.5.0",
            type="LSTM",
            status="deprecated",
            created_at=datetime.utcnow() - timedelta(days=7),
            metrics={
                "mae": 178.3,
                "rmse": 245.1,
                "mape": 12.1
            }
        )
    ]
    
    logger.info("Models list requested",
               user_id=current_user.get("user_id"),
               models_count=len(models))
    
    return models


@api_router.post("/upload",
                 tags=["Data"],
                 summary="Upload Data",
                 description="Upload CSV data for processing")
async def upload_data(
    file: UploadFile = File(...),
    current_user: Dict = Depends(get_current_user)
):
    """
    Upload CSV data file
    
    Args:
        file: Uploaded CSV file
        current_user: Authenticated user information
    
    Returns:
        Upload confirmation and data summary
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    try:
        # Read CSV data
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Basic validation
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        # Data summary
        summary = {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "upload_timestamp": datetime.utcnow()
        }
        
        logger.info("Data file uploaded successfully",
                   user_id=current_user.get("user_id"),
                   filename=file.filename,
                   rows=len(df),
                   columns=len(df.columns))
        
        return {
            "message": "File uploaded successfully",
            "summary": summary
        }
        
    except Exception as e:
        logger.error("File upload failed",
                    user_id=current_user.get("user_id"),
                    filename=file.filename,
                    error=str(e))
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process uploaded file: {str(e)}"
        )


# Background task functions

async def generate_predictions_async(processed_data: Dict, forecast_horizon: int) -> List[float]:
    """
    Generate predictions asynchronously (simplified implementation)
    
    Args:
        processed_data: Processed data from LSTM processor
        forecast_horizon: Number of periods to forecast
    
    Returns:
        List of predicted values
    """
    # Simulate async model inference
    await asyncio.sleep(0.1)
    
    # Generate dummy predictions for demo
    last_value = 1000.0  # This would come from actual model
    predictions = []
    
    for i in range(forecast_horizon):
        # Simple random walk for demo
        next_value = last_value * (1 + np.random.normal(0, 0.02))
        predictions.append(round(next_value, 2))
        last_value = next_value
    
    return predictions


async def process_single_prediction(
    request: PredictionRequest,
    current_user: Dict
) -> PredictionResponse:
    """
    Process a single prediction request
    
    Args:
        request: Prediction request
        current_user: User information
    
    Returns:
        Prediction response
    """
    # Convert to DataFrame
    df = pd.DataFrame(request.historical_data)
    
    # Generate predictions
    predictions = await generate_predictions_async({}, request.forecast_horizon)
    
    return PredictionResponse(
        predictions=predictions,
        confidence_intervals=[
            {"lower": p * 0.9, "upper": p * 1.1} 
            for p in predictions
        ],
        metadata={
            "model_version": "2.0.0",
            "sequence_length": request.sequence_length,
            "forecast_horizon": request.forecast_horizon
        }
    )


async def log_prediction_audit(
    user_id: str,
    request: PredictionRequest,
    predictions: List[float]
):
    """
    Log prediction for audit purposes
    
    Args:
        user_id: User identifier
        request: Original prediction request
        predictions: Generated predictions
    """
    logger.info("Prediction audit log",
               user_id=user_id,
               data_points=len(request.historical_data),
               predictions_count=len(predictions),
               timestamp=datetime.utcnow())


async def start_training_job(
    job_id: str,
    request: TrainingRequest,
    user_id: str
):
    """
    Start model training job
    
    Args:
        job_id: Training job identifier
        request: Training request
        user_id: User identifier
    """
    logger.info("Training job started",
               job_id=job_id,
               user_id=user_id,
               config=request.config)
    
    try:
        # Simulate training process
        await asyncio.sleep(5)  # Simulate training time
        
        logger.info("Training job completed successfully",
                   job_id=job_id,
                   user_id=user_id)
        
    except Exception as e:
        logger.error("Training job failed",
                    job_id=job_id,
                    user_id=user_id,
                    error=str(e))


# Include router in main application
__all__ = ["api_router"]