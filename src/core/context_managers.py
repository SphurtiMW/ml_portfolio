"""
Advanced Context Managers Module

This module provides professional context managers for:
- Resource management and cleanup
- Database connection pooling
- Model lifecycle management
- File operations with error handling
- Async context managers
- Performance monitoring contexts
- Transaction management
"""

import asyncio
import time
import tempfile
from contextlib import contextmanager, asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Dict, AsyncGenerator, Generator, Union
import threading
from datetime import datetime
import pickle
import json

try:
    import psycopg2
    from psycopg2 import pool
except ImportError:
    psycopg2 = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

from .config import settings
from .logger import get_logger, performance_logger
from .exceptions import handle_error, ModelError, DataError

logger = get_logger(__name__)


class DatabaseConnection:
    """Professional database connection context manager with pooling"""
    
    _pool = None
    _lock = threading.Lock()
    
    def __init__(self, connection_string: Optional[str] = None, autocommit: bool = False):
        self.connection_string = connection_string or settings.database.url
        self.autocommit = autocommit
        self.connection = None
        self.cursor = None
    
    @classmethod
    def init_pool(cls, min_conn: int = 1, max_conn: int = 10):
        """Initialize connection pool"""
        if psycopg2 is None:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        
        with cls._lock:
            if cls._pool is None:
                cls._pool = psycopg2.pool.ThreadedConnectionPool(
                    min_conn, max_conn, settings.database.url
                )
                logger.info("Database connection pool initialized",
                           min_connections=min_conn,
                           max_connections=max_conn)
    
    def __enter__(self):
        try:
            if self._pool is None:
                self.init_pool()
            
            # Get connection from pool
            self.connection = self._pool.getconn()
            if self.autocommit:
                self.connection.autocommit = True
            
            self.cursor = self.connection.cursor()
            
            logger.debug("Database connection established",
                        autocommit=self.autocommit,
                        connection_id=id(self.connection))
            
            return self
        
        except Exception as e:
            logger.error("Failed to establish database connection", error=str(e))
            raise handle_error(e)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None and not self.autocommit:
                # Commit transaction if no exception
                self.connection.commit()
                logger.debug("Database transaction committed")
            elif exc_type is not None and not self.autocommit:
                # Rollback on exception
                self.connection.rollback()
                logger.warning("Database transaction rolled back", 
                             exception_type=str(exc_type),
                             exception_value=str(exc_val))
            
            if self.cursor:
                self.cursor.close()
            
            if self.connection and self._pool:
                self._pool.putconn(self.connection)
                logger.debug("Database connection returned to pool")
        
        except Exception as e:
            logger.error("Error during database connection cleanup", error=str(e))
    
    def execute(self, query: str, params: tuple = None):
        """Execute a query"""
        if not self.cursor:
            raise RuntimeError("No active database cursor")
        
        try:
            self.cursor.execute(query, params)
            logger.debug("Query executed", query=query[:100])
        except Exception as e:
            logger.error("Query execution failed", query=query[:100], error=str(e))
            raise handle_error(e)
    
    def fetchone(self):
        """Fetch one row"""
        return self.cursor.fetchone()
    
    def fetchall(self):
        """Fetch all rows"""
        return self.cursor.fetchall()
    
    def fetchmany(self, size: int):
        """Fetch many rows"""
        return self.cursor.fetchmany(size)


class ModelManager:
    """Context manager for ML model lifecycle management"""
    
    def __init__(self, model_path: Union[str, Path], load_on_enter: bool = True):
        self.model_path = Path(model_path)
        self.load_on_enter = load_on_enter
        self.model = None
        self._temp_files = []
        self._start_time = None
    
    def __enter__(self):
        self._start_time = time.perf_counter()
        
        try:
            if self.load_on_enter and self.model_path.exists():
                self.model = self.load_model()
                logger.info("Model loaded successfully",
                           model_path=str(self.model_path),
                           model_type=type(self.model).__name__)
            
            return self
        
        except Exception as e:
            logger.error("Failed to load model", 
                        model_path=str(self.model_path),
                        error=str(e))
            raise ModelError(f"Model loading failed: {str(e)}", cause=e)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # Cleanup temporary files
            for temp_file in self._temp_files:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug("Temporary file cleaned up", file=str(temp_file))
            
            # Clear model from memory if needed
            if self.model and hasattr(self.model, 'clear_session'):
                self.model.clear_session()
            
            # Log session duration
            if self._start_time:
                duration = time.perf_counter() - self._start_time
                logger.info("Model session completed",
                           duration_seconds=duration,
                           success=exc_type is None)
        
        except Exception as e:
            logger.error("Error during model cleanup", error=str(e))
    
    def load_model(self):
        """Load model from file"""
        try:
            if self.model_path.suffix == '.h5' and tf is not None:
                # TensorFlow/Keras model
                return tf.keras.models.load_model(self.model_path)
            elif self.model_path.suffix == '.pkl':
                # Pickle model
                with open(self.model_path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported model format: {self.model_path.suffix}")
        
        except Exception as e:
            raise ModelError(f"Failed to load model from {self.model_path}: {str(e)}", cause=e)
    
    def save_model(self, model: Any, path: Optional[Path] = None):
        """Save model to file"""
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(model, tf.keras.Model):
                model.save(save_path)
            else:
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info("Model saved successfully", model_path=str(save_path))
        
        except Exception as e:
            raise ModelError(f"Failed to save model to {save_path}: {str(e)}", cause=e)
    
    def create_temp_file(self, suffix: str = '.tmp') -> Path:
        """Create a temporary file and track it for cleanup"""
        temp_file = Path(tempfile.mktemp(suffix=suffix))
        self._temp_files.append(temp_file)
        return temp_file


@contextmanager
def file_operations(file_path: Union[str, Path], 
                   mode: str = 'r',
                   encoding: str = 'utf-8',
                   backup: bool = False) -> Generator:
    """
    Context manager for safe file operations with backup and error handling
    
    Args:
        file_path: Path to the file
        mode: File open mode
        encoding: File encoding
        backup: Whether to create backup before writing
    
    Yields:
        File object
    """
    file_path = Path(file_path)
    backup_path = None
    
    try:
        # Create backup if requested and file exists
        if backup and 'w' in mode and file_path.exists():
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            backup_path.write_bytes(file_path.read_bytes())
            logger.debug("Backup created", original=str(file_path), backup=str(backup_path))
        
        # Ensure parent directory exists for write operations
        if 'w' in mode or 'a' in mode:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, mode, encoding=encoding) as file:
            yield file
            
        logger.debug("File operation completed", file_path=str(file_path), mode=mode)
    
    except Exception as e:
        logger.error("File operation failed", 
                    file_path=str(file_path),
                    mode=mode,
                    error=str(e))
        
        # Restore backup if write operation failed
        if backup_path and backup_path.exists() and 'w' in mode:
            try:
                file_path.write_bytes(backup_path.read_bytes())
                logger.info("File restored from backup", file_path=str(file_path))
            except Exception as restore_error:
                logger.error("Failed to restore from backup", error=str(restore_error))
        
        raise DataError(f"File operation failed: {str(e)}", cause=e)
    
    finally:
        # Clean up backup file if operation was successful
        if backup_path and backup_path.exists():
            try:
                backup_path.unlink()
                logger.debug("Backup file cleaned up", backup_path=str(backup_path))
            except Exception:
                pass  # Ignore cleanup errors


@contextmanager
def performance_monitor(operation_name: str, **context) -> Generator:
    """
    Context manager for monitoring operation performance
    
    Args:
        operation_name: Name of the operation being monitored
        **context: Additional context for logging
    
    Yields:
        Performance data dictionary
    """
    perf_data = {
        'operation': operation_name,
        'start_time': time.perf_counter(),
        'start_memory': None,
        **context
    }
    
    try:
        # Try to get memory usage
        import psutil
        process = psutil.Process()
        perf_data['start_memory'] = process.memory_info().rss
    except ImportError:
        pass
    
    logger.info(f"Performance monitoring started: {operation_name}", **context)
    
    try:
        yield perf_data
        
        # Calculate performance metrics
        end_time = time.perf_counter()
        duration = end_time - perf_data['start_time']
        
        perf_data.update({
            'end_time': end_time,
            'duration_seconds': duration,
            'status': 'success'
        })
        
        try:
            import psutil
            process = psutil.Process()
            end_memory = process.memory_info().rss
            if perf_data['start_memory']:
                perf_data['memory_delta_mb'] = (end_memory - perf_data['start_memory']) / 1024 / 1024
        except ImportError:
            pass
        
        logger.info(f"Performance monitoring completed: {operation_name}",
                   **{k: v for k, v in perf_data.items() if k != 'operation'})
    
    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - perf_data['start_time']
        
        perf_data.update({
            'end_time': end_time,
            'duration_seconds': duration,
            'status': 'error',
            'error': str(e)
        })
        
        logger.error(f"Performance monitoring failed: {operation_name}",
                    **{k: v for k, v in perf_data.items() if k != 'operation'})
        raise


@contextmanager
def temporary_directory(prefix: str = "forecast_", cleanup: bool = True) -> Generator[Path, None, None]:
    """
    Context manager for temporary directory operations
    
    Args:
        prefix: Prefix for temporary directory name
        cleanup: Whether to clean up directory on exit
    
    Yields:
        Path to temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    
    try:
        logger.debug("Temporary directory created", temp_dir=str(temp_dir))
        yield temp_dir
    
    finally:
        if cleanup:
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug("Temporary directory cleaned up", temp_dir=str(temp_dir))
            except Exception as e:
                logger.warning("Failed to cleanup temporary directory",
                             temp_dir=str(temp_dir),
                             error=str(e))


@contextmanager
def json_config(config_path: Union[str, Path], 
               create_if_missing: bool = True,
               default_config: Optional[Dict] = None) -> Generator[Dict, None, None]:
    """
    Context manager for JSON configuration file operations
    
    Args:
        config_path: Path to configuration file
        create_if_missing: Whether to create file if it doesn't exist
        default_config: Default configuration to use if file doesn't exist
    
    Yields:
        Configuration dictionary
    """
    config_path = Path(config_path)
    config = default_config or {}
    
    try:
        # Load existing config
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.debug("Configuration loaded", config_path=str(config_path))
        elif create_if_missing:
            # Create directory if needed
            config_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Configuration will be created", config_path=str(config_path))
        
        yield config
        
        # Save config back to file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            logger.debug("Configuration saved", config_path=str(config_path))
    
    except Exception as e:
        logger.error("Configuration operation failed",
                    config_path=str(config_path),
                    error=str(e))
        raise DataError(f"Configuration operation failed: {str(e)}", cause=e)


# Async context managers

@asynccontextmanager
async def async_file_operations(file_path: Union[str, Path], 
                               mode: str = 'r',
                               encoding: str = 'utf-8') -> AsyncGenerator:
    """
    Async context manager for file operations
    
    Args:
        file_path: Path to the file
        mode: File open mode
        encoding: File encoding
    
    Yields:
        Async file object
    """
    import aiofiles
    
    file_path = Path(file_path)
    
    try:
        # Ensure parent directory exists for write operations
        if 'w' in mode or 'a' in mode:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, mode, encoding=encoding) as file:
            logger.debug("Async file operation started", 
                        file_path=str(file_path), mode=mode)
            yield file
            
        logger.debug("Async file operation completed", 
                    file_path=str(file_path), mode=mode)
    
    except Exception as e:
        logger.error("Async file operation failed", 
                    file_path=str(file_path),
                    mode=mode,
                    error=str(e))
        raise DataError(f"Async file operation failed: {str(e)}", cause=e)


@asynccontextmanager
async def async_performance_monitor(operation_name: str, **context) -> AsyncGenerator:
    """
    Async context manager for monitoring operation performance
    
    Args:
        operation_name: Name of the operation being monitored
        **context: Additional context for logging
    
    Yields:
        Performance data dictionary
    """
    perf_data = {
        'operation': operation_name,
        'start_time': time.perf_counter(),
        **context
    }
    
    logger.info(f"Async performance monitoring started: {operation_name}", **context)
    
    try:
        yield perf_data
        
        # Calculate performance metrics
        end_time = time.perf_counter()
        duration = end_time - perf_data['start_time']
        
        perf_data.update({
            'end_time': end_time,
            'duration_seconds': duration,
            'status': 'success'
        })
        
        logger.info(f"Async performance monitoring completed: {operation_name}",
                   **{k: v for k, v in perf_data.items() if k != 'operation'})
    
    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - perf_data['start_time']
        
        perf_data.update({
            'end_time': end_time,
            'duration_seconds': duration,
            'status': 'error',
            'error': str(e)
        })
        
        logger.error(f"Async performance monitoring failed: {operation_name}",
                    **{k: v for k, v in perf_data.items() if k != 'operation'})
        raise


@contextmanager
def resource_pool(resource_factory, max_size: int = 10, timeout: float = 30.0):
    """
    Context manager for managing a pool of resources
    
    Args:
        resource_factory: Function to create new resources
        max_size: Maximum pool size
        timeout: Timeout for acquiring resource
    
    Yields:
        Resource from pool
    """
    import queue
    import threading
    
    pool = queue.Queue(maxsize=max_size)
    created_resources = []
    lock = threading.Lock()
    
    def get_resource():
        try:
            # Try to get existing resource
            return pool.get_nowait()
        except queue.Empty:
            # Create new resource if pool is empty
            with lock:
                if len(created_resources) < max_size:
                    resource = resource_factory()
                    created_resources.append(resource)
                    return resource
                else:
                    # Wait for resource to become available
                    return pool.get(timeout=timeout)
    
    def return_resource(resource):
        try:
            pool.put_nowait(resource)
        except queue.Full:
            pass  # Pool is full, resource will be garbage collected
    
    resource = None
    try:
        resource = get_resource()
        logger.debug("Resource acquired from pool")
        yield resource
    
    finally:
        if resource:
            return_resource(resource)
            logger.debug("Resource returned to pool")


# Export context managers
__all__ = [
    'DatabaseConnection',
    'ModelManager',
    'file_operations',
    'performance_monitor',
    'temporary_directory',
    'json_config',
    'async_file_operations',
    'async_performance_monitor',
    'resource_pool'
]