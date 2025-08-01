"""
Advanced Configuration Management System

This module provides:
- Environment-based configuration
- Type validation with Pydantic
- Secrets management
- Database configuration
- MLOps settings
- API configuration
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from pydantic.networks import AnyHttpUrl, PostgresDsn
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    user: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="password", env="DB_PASSWORD")
    database: str = Field(default="forecasting_db", env="DB_NAME")
    
    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "DB_"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and task queue"""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_prefix = "REDIS_"


class MLFlowSettings(BaseSettings):
    """MLFlow configuration for experiment tracking"""
    
    tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(default="demand_forecasting", env="MLFLOW_EXPERIMENT_NAME")
    model_registry_uri: Optional[str] = Field(default=None, env="MLFLOW_MODEL_REGISTRY_URI")
    
    class Config:
        env_prefix = "MLFLOW_"


class ModelSettings(BaseSettings):
    """Model training and prediction settings"""
    
    # Data preprocessing
    sequence_length: int = Field(default=60, description="LSTM sequence length")
    train_test_split: float = Field(default=0.8, ge=0.1, le=0.9)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.3)
    
    # Model architecture
    lstm_units_1: int = Field(default=64, ge=8, le=512)
    lstm_units_2: int = Field(default=32, ge=8, le=256)
    dense_units: int = Field(default=16, ge=4, le=128)
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.5)
    
    # Training parameters
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)
    batch_size: int = Field(default=32, ge=8, le=512)
    epochs: int = Field(default=100, ge=10, le=1000)
    early_stopping_patience: int = Field(default=10, ge=5, le=50)
    
    # Model management
    model_save_path: Path = Field(default=Path("artifacts/models"))
    max_model_versions: int = Field(default=5, ge=1, le=20)
    retrain_threshold_mae: float = Field(default=1000.0, ge=0.0)
    retrain_schedule_hours: int = Field(default=24, ge=1, le=168)
    
    @validator('model_save_path')
    def create_model_path(cls, v):
        """Ensure model save directory exists"""
        v.mkdir(parents=True, exist_ok=True)
        return v


class APISettings(BaseSettings):
    """API and web service configuration"""
    
    title: str = "Demand Forecasting API"
    description: str = "Professional ML API for time series demand forecasting"
    version: str = "2.0.0"
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    
    # CORS
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    allowed_methods: List[str] = Field(default=["*"], env="ALLOWED_METHODS")
    allowed_headers: List[str] = Field(default=["*"], env="ALLOWED_HEADERS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # seconds
    
    class Config:
        env_prefix = "API_"


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    file_enabled: bool = Field(default=True, env="LOG_FILE_ENABLED")
    file_path: Path = Field(default=Path("logs"), env="LOG_FILE_PATH")
    max_file_size: int = Field(default=10485760, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Structured logging
    structured: bool = Field(default=True, env="LOG_STRUCTURED")
    json_format: bool = Field(default=True, env="LOG_JSON_FORMAT")
    
    @validator('file_path')
    def create_log_path(cls, v):
        """Ensure log directory exists"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_prefix = "LOG_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration"""
    
    # Prometheus metrics
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    
    # Health checks
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    model_health_threshold: float = Field(default=0.95, env="MODEL_HEALTH_THRESHOLD")
    
    # Alerting
    alert_webhook_url: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    alert_email: Optional[str] = Field(default=None, env="ALERT_EMAIL")
    
    class Config:
        env_prefix = "MONITORING_"


class Settings(BaseSettings):
    """Main application settings combining all configuration modules"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    project_name: str = Field(default="demand-forecasting", env="PROJECT_NAME")
    project_root: Path = Field(default=Path(__file__).parent.parent.parent)
    
    # Sub-configurations
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    mlflow: MLFlowSettings = MLFlowSettings()
    model: ModelSettings = ModelSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    # Data paths
    data_path: Path = Field(default=Path("data"))
    artifacts_path: Path = Field(default=Path("artifacts"))
    
    @validator('data_path', 'artifacts_path')
    def create_paths(cls, v):
        """Ensure required directories exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()


# Export environment-specific configurations
def get_database_url() -> str:
    """Get database connection URL"""
    return settings.database.url


def get_redis_url() -> str:
    """Get Redis connection URL"""
    return settings.redis.url


def get_model_config() -> Dict[str, Any]:
    """Get model configuration as dictionary"""
    return settings.model.dict()


def get_api_config() -> Dict[str, Any]:
    """Get API configuration as dictionary"""
    return settings.api.dict()