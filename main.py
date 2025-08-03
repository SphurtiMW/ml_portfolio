#!/usr/bin/env python3
"""
Main entry point for the Demand Forecasting API

This script provides a simple way to start the application with different modes:
- API server (default)
- Data processing
- Model training
- Testing

Usage:
    python main.py                    # Start API server
    python main.py --mode api         # Start API server
    python main.py --mode train       # Run model training
    python main.py --mode process     # Run data processing
    python main.py --mode test        # Run tests
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.logger import get_logger, setup_logging
from src.core.config import settings

# Setup logging
setup_logging()
logger = get_logger(__name__)


def run_api_server():
    """Run the FastAPI server"""
    logger.info("Starting API server", 
                host=settings.api.host, 
                port=settings.api.port,
                environment=settings.environment)
    
    try:
        import uvicorn
        uvicorn.run(
            "src.api.app:app",
            host=settings.api.host,
            port=settings.api.port,
            reload=settings.api.debug,
            log_level=settings.logging.level.lower(),
            access_log=True,
            loop="asyncio"
        )
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)


def run_data_processing():
    """Run data processing example"""
    logger.info("Running data processing example")
    
    try:
        import pandas as pd
        from src.data.processors import DataProcessorFactory, LoggingObserver
        
        # Check if data file exists
        data_file = Path("artifacts/Historical Product Demand.csv")
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            logger.info("Please ensure the Historical Product Demand.csv file is in the artifacts/ directory")
            return
        
        # Load data
        df = pd.read_csv(data_file, parse_dates=["Date"], index_col="Date")
        logger.info(f"Loaded data with {len(df)} rows")
        
        # Create processor
        processor = DataProcessorFactory.create_processor(
            'lstm',
            sequence_length=60,
            target_column='Order_Demand'
        )
        
        # Add observer
        observer = LoggingObserver()
        processor.add_observer(observer)
        
        # Process data
        result = processor.process(df)
        
        if result.success:
            logger.info("Data processing completed successfully",
                       processing_time=result.processing_time,
                       train_samples=len(result.data.get('X_train', [])),
                       test_samples=len(result.data.get('X_test', [])))
        else:
            logger.error("Data processing failed", errors=result.errors)
            
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Install dependencies with: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Data processing failed: {e}")


def run_eda_analysis():
    """Run comprehensive EDA analysis"""
    logger.info("Running comprehensive EDA analysis")
    
    try:
        from src.analysis.eda import run_eda_analysis
        
        # Check if data file exists
        data_file = Path("artifacts/Historical Product Demand.csv")
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            logger.info("Please ensure the Historical Product Demand.csv file is in the artifacts/ directory")
            return
        
        # Run comprehensive EDA
        result = run_eda_analysis(str(data_file))
        
        logger.info("EDA analysis completed successfully",
                   dataset_shape=result.dataset_info.get('shape'),
                   missing_data_percentage=result.missing_data_analysis.get('missing_percentage'),
                   outlier_percentage=result.outlier_analysis.get('outlier_percentage'))
        
        print("\n📊 EDA Analysis Summary:")
        print(f"Dataset Shape: {result.dataset_info.get('shape')}")
        print(f"Missing Data: {result.missing_data_analysis.get('missing_percentage'):.2f}%")
        print(f"Outliers: {result.outlier_analysis.get('outlier_percentage', 0):.2f}%")
        print("Check the 'plots' directory for visualizations!")
        
    except ImportError as e:
        logger.error(f"Missing dependencies for EDA: {e}")
        logger.info("Install dependencies with: python3 install_demo.py")
    except Exception as e:
        logger.error(f"EDA analysis failed: {e}")


def run_model_training():
    """Run model training example"""
    logger.info("Running model training example")
    
    try:
        # First run data processing
        run_data_processing()
        
        logger.info("Model training would start here...")
        logger.info("This is a placeholder for the actual training implementation")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")


def run_tests():
    """Run tests"""
    logger.info("Running tests")
    
    try:
        import pytest
        
        # Run tests with coverage
        exit_code = pytest.main([
            "tests/",
            "-v",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html"
        ])
        
        if exit_code == 0:
            logger.info("All tests passed!")
        else:
            logger.error("Some tests failed")
            
    except ImportError:
        logger.error("pytest not installed. Install with: pip install pytest pytest-cov")
        sys.exit(1)


def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import pydantic
    except ImportError as e:
        missing_deps.append(str(e))
    
    if missing_deps:
        logger.error("Missing dependencies:", errors=missing_deps)
        logger.info("Install dependencies with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Demand Forecasting API - Professional ML Application"
    )
    
    parser.add_argument(
        "--mode",
        choices=["api", "process", "train", "test", "eda"],
        default="api",
        help="Application mode (default: api)"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if dependencies are installed"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 60)
    print("🚀 Professional Demand Forecasting API v2.0.0")
    print("=" * 60)
    
    # Check dependencies if requested
    if args.check_deps:
        if check_dependencies():
            print("✅ All dependencies are installed")
        else:
            print("❌ Missing dependencies")
            sys.exit(1)
        return
    
    # Check basic dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run based on mode
    try:
        if args.mode == "api":
            print(f"🌐 Starting API server on http://{settings.api.host}:{settings.api.port}")
            print(f"📚 API docs available at http://{settings.api.host}:{settings.api.port}/docs")
            print(f"📊 Metrics available at http://{settings.api.host}:{settings.monitoring.prometheus_port}/metrics")
            print("=" * 60)
            run_api_server()
            
        elif args.mode == "process":
            print("📊 Running data processing example")
            print("=" * 60)
            run_data_processing()
            
        elif args.mode == "train":
            print("🤖 Running model training example")
            print("=" * 60)
            run_model_training()
            
        elif args.mode == "test":
            print("🧪 Running tests")
            print("=" * 60)
            run_tests()
            
        elif args.mode == "eda":
            print("📊 Running comprehensive EDA analysis")
            print("=" * 60)
            run_eda_analysis()
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()