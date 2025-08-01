"""
Professional Data Processors Module

This module implements data processing using advanced OOP patterns:
- Abstract base classes and inheritance
- Strategy pattern for different processing algorithms
- Factory pattern for processor creation
- Async processing for I/O bound operations
- Decorator pattern for processing enhancement
- Observer pattern for progress monitoring
"""

import asyncio
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator, Protocol
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from pathlib import Path

from ..core.logger import get_logger
from ..core.decorators import timer, retry, cache_result
from ..core.context_managers import performance_monitor, async_performance_monitor
from ..core.exceptions import DataError, DataValidationError
from ..core.config import settings

logger = get_logger(__name__)


class ProcessingStrategy(Enum):
    """Data processing strategies"""
    BATCH = "batch"
    STREAMING = "streaming"
    ASYNC_BATCH = "async_batch"
    DISTRIBUTED = "distributed"


@dataclass
class ProcessingResult:
    """Result of data processing operation"""
    data: Any
    metadata: Dict[str, Any]
    processing_time: float
    strategy: ProcessingStrategy
    success: bool = True
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ProcessingObserver(Protocol):
    """Observer interface for processing progress"""
    
    def on_start(self, total_items: int) -> None:
        """Called when processing starts"""
        ...
    
    def on_progress(self, current: int, total: int, item_data: Any = None) -> None:
        """Called during processing progress"""
        ...
    
    def on_complete(self, result: ProcessingResult) -> None:
        """Called when processing completes"""
        ...
    
    def on_error(self, error: Exception, item_data: Any = None) -> None:
        """Called when an error occurs"""
        ...


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processors using Template Method pattern
    """
    
    def __init__(self, strategy: ProcessingStrategy = ProcessingStrategy.BATCH):
        self.strategy = strategy
        self._observers: List[ProcessingObserver] = []
        self._config = settings.model.dict()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    def add_observer(self, observer: ProcessingObserver) -> None:
        """Add processing observer"""
        self._observers.append(observer)
    
    def remove_observer(self, observer: ProcessingObserver) -> None:
        """Remove processing observer"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_start(self, total_items: int) -> None:
        """Notify observers of processing start"""
        for observer in self._observers:
            try:
                observer.on_start(total_items)
            except Exception as e:
                self.logger.warning(f"Observer notification failed: {e}")
    
    def _notify_progress(self, current: int, total: int, item_data: Any = None) -> None:
        """Notify observers of progress"""
        for observer in self._observers:
            try:
                observer.on_progress(current, total, item_data)
            except Exception as e:
                self.logger.warning(f"Observer notification failed: {e}")
    
    def _notify_complete(self, result: ProcessingResult) -> None:
        """Notify observers of completion"""
        for observer in self._observers:
            try:
                observer.on_complete(result)
            except Exception as e:
                self.logger.warning(f"Observer notification failed: {e}")
    
    def _notify_error(self, error: Exception, item_data: Any = None) -> None:
        """Notify observers of error"""
        for observer in self._observers:
            try:
                observer.on_error(error, item_data)
            except Exception as e:
                self.logger.warning(f"Observer notification failed: {e}")
    
    @abstractmethod
    def _validate_input(self, data: Any) -> bool:
        """Validate input data"""
        pass
    
    @abstractmethod
    def _preprocess(self, data: Any) -> Any:
        """Preprocess data"""
        pass
    
    @abstractmethod
    def _process_core(self, data: Any) -> Any:
        """Core processing logic"""
        pass
    
    @abstractmethod
    def _postprocess(self, data: Any) -> Any:
        """Postprocess data"""
        pass
    
    @timer
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """
        Template method for data processing
        
        Args:
            data: Input data to process
            **kwargs: Additional processing parameters
        
        Returns:
            ProcessingResult with processed data and metadata
        """
        start_time = asyncio.get_event_loop().time() if asyncio.iscoroutinefunction(self._process_core) else pd.Timestamp.now().timestamp()
        errors = []
        
        try:
            self.logger.info("Starting data processing",
                           strategy=self.strategy.value,
                           data_type=type(data).__name__)
            
            # Notify observers
            total_items = len(data) if hasattr(data, '__len__') else 1
            self._notify_start(total_items)
            
            # Template method steps
            if not self._validate_input(data):
                raise DataValidationError("data", type(data), "valid input format")
            
            preprocessed_data = self._preprocess(data)
            processed_data = self._process_core(preprocessed_data, **kwargs)
            final_data = self._postprocess(processed_data)
            
            # Calculate processing time
            end_time = asyncio.get_event_loop().time() if asyncio.iscoroutinefunction(self._process_core) else pd.Timestamp.now().timestamp()
            processing_time = end_time - start_time
            
            # Create result
            result = ProcessingResult(
                data=final_data,
                metadata={
                    "strategy": self.strategy.value,
                    "processor": self.__class__.__name__,
                    "input_shape": getattr(data, 'shape', None),
                    "output_shape": getattr(final_data, 'shape', None),
                    "config": self._config,
                    **kwargs
                },
                processing_time=processing_time,
                strategy=self.strategy,
                success=True,
                errors=errors
            )
            
            self._notify_complete(result)
            
            self.logger.info("Data processing completed successfully",
                           processing_time=processing_time,
                           strategy=self.strategy.value)
            
            return result
        
        except Exception as e:
            self._notify_error(e, data)
            error_msg = f"Data processing failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Return failed result
            end_time = asyncio.get_event_loop().time() if asyncio.iscoroutinefunction(self._process_core) else pd.Timestamp.now().timestamp()
            
            return ProcessingResult(
                data=None,
                metadata={"error": str(e)},
                processing_time=end_time - start_time,
                strategy=self.strategy,
                success=False,
                errors=[error_msg]
            )


class LSTMDataProcessor(BaseDataProcessor):
    """
    Specialized processor for LSTM time series data
    Implements Chain of Responsibility pattern for processing steps
    """
    
    def __init__(self, 
                 sequence_length: int = None,
                 target_column: str = 'Order_Demand',
                 scaling_strategy: str = 'minmax',
                 **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length or settings.model.sequence_length
        self.target_column = target_column
        self.scaling_strategy = scaling_strategy
        self._scaler = None
        self._feature_columns = []
    
    def _validate_input(self, data: Any) -> bool:
        """Validate LSTM input data"""
        if not isinstance(data, pd.DataFrame):
            return False
        
        if self.target_column not in data.columns:
            self.logger.error(f"Target column '{self.target_column}' not found in data")
            return False
        
        if len(data) < self.sequence_length:
            self.logger.error(f"Data length {len(data)} is less than sequence length {self.sequence_length}")
            return False
        
        return True
    
    @cache_result(ttl=3600)  # Cache preprocessing for 1 hour
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for LSTM"""
        self.logger.debug("Preprocessing LSTM data")
        
        processed_data = data.copy()
        
        # Handle missing values
        if processed_data.isnull().any().any():
            self.logger.warning("Found missing values, applying forward fill")
            processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
        
        # Apply log transformation to target column
        if self.target_column in processed_data.columns:
            processed_data[f'{self.target_column}_log'] = np.log1p(processed_data[self.target_column])
            self.logger.debug("Applied log transformation to target column")
        
        # Feature engineering
        processed_data = self._engineer_features(processed_data)
        
        return processed_data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for time series"""
        self.logger.debug("Engineering time series features")
        
        # Create date-based features if index is datetime
        if isinstance(data.index, pd.DatetimeIndex):
            data['day_of_week'] = data.index.dayofweek
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
        
        # Create lag features
        for lag in [1, 7, 30]:
            if f'{self.target_column}_log' in data.columns:
                data[f'{self.target_column}_lag_{lag}'] = data[f'{self.target_column}_log'].shift(lag)
        
        # Rolling statistics
        if f'{self.target_column}_log' in data.columns:
            data[f'{self.target_column}_roll_mean_7'] = data[f'{self.target_column}_log'].rolling(window=7).mean()
            data[f'{self.target_column}_roll_std_7'] = data[f'{self.target_column}_log'].rolling(window=7).std()
        
        # Drop rows with NaN values created by feature engineering
        data = data.dropna()
        
        return data
    
    def _process_core(self, data: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Core LSTM data processing"""
        self.logger.debug("Processing LSTM sequences")
        
        # Select features for scaling
        target_col = f'{self.target_column}_log'
        feature_cols = [col for col in data.columns if col != self.target_column]
        
        # Scale features
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        
        if self.scaling_strategy == 'minmax':
            self._scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self._scaler = StandardScaler()
        
        # Fit scaler on training portion (first 80%)
        train_size = int(len(data) * settings.model.train_test_split)
        train_data = data.iloc[:train_size]
        
        self._scaler.fit(train_data[[target_col]])
        scaled_data = self._scaler.transform(data[[target_col]])
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        self.logger.info("LSTM sequences created",
                        input_shape=X.shape,
                        output_shape=y.shape,
                        sequence_length=self.sequence_length)
        
        return X, y
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
            
            # Notify progress
            if i % 1000 == 0:
                self._notify_progress(i, len(data) - self.sequence_length)
        
        return np.array(X), np.array(y)
    
    def _postprocess(self, data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
        """Postprocess LSTM data"""
        X, y = data
        
        # Split into train/test
        train_size = int(len(X) * settings.model.train_test_split)
        
        result = {
            'X_train': X[:train_size],
            'y_train': y[:train_size],
            'X_test': X[train_size:],
            'y_test': y[train_size:],
            'scaler': self._scaler
        }
        
        self.logger.info("LSTM data postprocessing completed",
                        train_samples=len(result['X_train']),
                        test_samples=len(result['X_test']))
        
        return result
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data"""
        if self._scaler is None:
            raise ValueError("Scaler not fitted. Process data first.")
        
        return self._scaler.inverse_transform(scaled_data.reshape(-1, 1))


class AsyncDataProcessor(BaseDataProcessor):
    """
    Async data processor for I/O bound operations
    Implements Producer-Consumer pattern with async queues
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 batch_size: int = 1000,
                 **kwargs):
        super().__init__(ProcessingStrategy.ASYNC_BATCH, **kwargs)
        self.max_workers = max_workers
        self.batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_workers)
    
    def _validate_input(self, data: Any) -> bool:
        """Validate async input data"""
        return hasattr(data, '__iter__') or isinstance(data, (list, tuple, pd.DataFrame))
    
    async def _preprocess_async(self, data: Any) -> Any:
        """Async preprocessing"""
        self.logger.debug("Async preprocessing started")
        
        if isinstance(data, pd.DataFrame):
            # Async CSV processing simulation
            await asyncio.sleep(0.01)  # Simulate I/O delay
            return data.copy()
        
        return data
    
    def _preprocess(self, data: Any) -> Any:
        """Sync wrapper for async preprocessing"""
        if asyncio.iscoroutinefunction(self._preprocess_async):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._preprocess_async(data))
        return data
    
    async def _process_batch_async(self, batch: Any) -> Any:
        """Process a single batch asynchronously"""
        async with self._semaphore:
            self.logger.debug(f"Processing batch of size {len(batch) if hasattr(batch, '__len__') else 'unknown'}")
            
            # Simulate async processing
            await asyncio.sleep(0.1)
            
            # Apply transformations
            if isinstance(batch, pd.DataFrame):
                return batch.apply(lambda x: x * 1.1 if x.dtype.kind in 'biufc' else x)
            
            return batch
    
    def _process_core(self, data: Any, **kwargs) -> Any:
        """Core async processing"""
        return asyncio.run(self._process_core_async(data, **kwargs))
    
    async def _process_core_async(self, data: Any, **kwargs) -> Any:
        """Async core processing with batching"""
        self.logger.info("Starting async batch processing",
                        max_workers=self.max_workers,
                        batch_size=self.batch_size)
        
        if isinstance(data, pd.DataFrame):
            # Process dataframe in batches
            batches = [data.iloc[i:i + self.batch_size] 
                      for i in range(0, len(data), self.batch_size)]
        else:
            # Process list/array in batches
            batches = [data[i:i + self.batch_size] 
                      for i in range(0, len(data), self.batch_size)]
        
        # Process batches concurrently
        processed_batches = await asyncio.gather(
            *[self._process_batch_async(batch) for batch in batches],
            return_exceptions=True
        )
        
        # Combine results
        if isinstance(data, pd.DataFrame):
            result = pd.concat([batch for batch in processed_batches 
                              if isinstance(batch, pd.DataFrame)], ignore_index=True)
        else:
            result = []
            for batch in processed_batches:
                if isinstance(batch, Exception):
                    self.logger.error(f"Batch processing failed: {batch}")
                    continue
                result.extend(batch)
        
        return result
    
    def _postprocess(self, data: Any) -> Any:
        """Postprocess async results"""
        self.logger.debug("Async postprocessing completed")
        return data


class DataProcessorFactory:
    """
    Factory for creating data processors
    Implements Factory Method pattern
    """
    
    _processors = {
        'lstm': LSTMDataProcessor,
        'async': AsyncDataProcessor,
        'base': BaseDataProcessor
    }
    
    @classmethod
    def create_processor(cls, 
                        processor_type: str,
                        strategy: ProcessingStrategy = ProcessingStrategy.BATCH,
                        **kwargs) -> BaseDataProcessor:
        """
        Create data processor instance
        
        Args:
            processor_type: Type of processor to create
            strategy: Processing strategy
            **kwargs: Additional parameters for processor
        
        Returns:
            Data processor instance
        """
        if processor_type not in cls._processors:
            raise ValueError(f"Unknown processor type: {processor_type}. "
                           f"Available types: {list(cls._processors.keys())}")
        
        processor_class = cls._processors[processor_type]
        
        if processor_type == 'async':
            return processor_class(strategy=strategy, **kwargs)
        elif processor_type == 'lstm':
            return processor_class(strategy=strategy, **kwargs)
        else:
            return processor_class(strategy=strategy, **kwargs)
    
    @classmethod
    def register_processor(cls, name: str, processor_class: type) -> None:
        """Register new processor type"""
        if not issubclass(processor_class, BaseDataProcessor):
            raise ValueError("Processor must inherit from BaseDataProcessor")
        
        cls._processors[name] = processor_class
        logger.info(f"Registered new processor type: {name}")
    
    @classmethod
    def list_processors(cls) -> List[str]:
        """List available processor types"""
        return list(cls._processors.keys())


# Progress observer implementation
class LoggingObserver:
    """Observer that logs processing progress"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.ProgressObserver")
    
    def on_start(self, total_items: int) -> None:
        self.logger.info(f"Processing started: {total_items} items")
    
    def on_progress(self, current: int, total: int, item_data: Any = None) -> None:
        progress = (current / total) * 100
        if current % (total // 10) == 0:  # Log every 10%
            self.logger.info(f"Progress: {progress:.1f}% ({current}/{total})")
    
    def on_complete(self, result: ProcessingResult) -> None:
        self.logger.info(f"Processing completed successfully in {result.processing_time:.2f}s")
    
    def on_error(self, error: Exception, item_data: Any = None) -> None:
        self.logger.error(f"Processing error: {error}")


# Utility functions
def create_data_generator(data: pd.DataFrame, batch_size: int = 1000):
    """
    Generator function for streaming data processing
    
    Args:
        data: DataFrame to stream
        batch_size: Size of each batch
    
    Yields:
        Data batches
    """
    for i in range(0, len(data), batch_size):
        yield data.iloc[i:i + batch_size]


async def async_data_generator(data: pd.DataFrame, batch_size: int = 1000):
    """
    Async generator for streaming data processing
    
    Args:
        data: DataFrame to stream
        batch_size: Size of each batch
    
    Yields:
        Data batches asynchronously
    """
    for i in range(0, len(data), batch_size):
        await asyncio.sleep(0)  # Yield control
        yield data.iloc[i:i + batch_size]