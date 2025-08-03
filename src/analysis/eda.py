"""
Professional Exploratory Data Analysis Module

This module provides comprehensive EDA capabilities using OOP design:
- Professional data loading and preprocessing
- Statistical analysis with proper error handling
- Time series analysis and decomposition
- Advanced visualizations with context managers
- Feature engineering and correlation analysis
- Integration with the professional logging and configuration system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# Statistical libraries
try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from pandas.plotting import autocorrelation_plot
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from ..core.logger import get_logger
from ..core.decorators import timer, log_method
from ..core.context_managers import performance_monitor
from ..core.exceptions import DataError, DataValidationError
from ..core.config import settings

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

# Set professional plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class EDAResult:
    """Container for EDA analysis results"""
    dataset_info: Dict[str, Any]
    statistical_summary: Dict[str, Any]
    missing_data_analysis: Dict[str, Any]
    outlier_analysis: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    time_series_analysis: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, Any]] = None


class BaseAnalyzer(ABC):
    """Abstract base class for data analyzers"""
    
    def __init__(self, save_plots: bool = True, plot_dir: Optional[Path] = None):
        self.save_plots = save_plots
        self.plot_dir = plot_dir or Path("plots")
        self.plot_dir.mkdir(exist_ok=True)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform analysis on the dataframe"""
        pass
    
    def _save_plot(self, filename: str) -> None:
        """Save current plot to file"""
        if self.save_plots:
            filepath = self.plot_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Plot saved: {filepath}")


class DataLoader:
    """Professional data loader with comprehensive validation"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.DataLoader")
    
    @timer
    @log_method(include_args=True)
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data with comprehensive validation and error handling
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Loaded and validated DataFrame
            
        Raises:
            DataError: If data loading or validation fails
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            # Load data based on file extension
            if filepath.suffix.lower() == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            else:
                raise DataError(f"Unsupported file format: {filepath.suffix}")
            
            if df.empty:
                raise DataError("Loaded dataset is empty")
            
            self.logger.info("Dataset loaded successfully",
                           filename=filepath.name,
                           rows=len(df),
                           columns=len(df.columns))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {filepath}: {str(e)}")
            raise DataError(f"Data loading failed: {str(e)}", cause=e)
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict()
        }


class DataPreprocessor:
    """Professional data preprocessing with comprehensive validation"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.DataPreprocessor")
    
    @timer
    @log_method()
    def preprocess_time_series_data(self, df: pd.DataFrame, 
                                  date_column: str = 'Date',
                                  target_column: str = 'Order_Demand') -> pd.DataFrame:
        """
        Preprocess time series data with comprehensive validation
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            target_column: Name of the target column
            
        Returns:
            Preprocessed DataFrame
            
        Raises:
            DataValidationError: If preprocessing validation fails
        """
        df = df.copy()
        
        try:
            # Validate required columns
            required_columns = [date_column, target_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataValidationError(
                    field="columns",
                    value=missing_columns,
                    expected=f"Required columns: {required_columns}"
                )
            
            # Convert date column to datetime
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            invalid_dates = df[date_column].isnull().sum()
            if invalid_dates > 0:
                self.logger.warning(f"Found {invalid_dates} invalid dates, dropping rows")
                df = df.dropna(subset=[date_column])
            
            # Convert target column to numeric
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
            invalid_values = df[target_column].isnull().sum()
            if invalid_values > 0:
                self.logger.warning(f"Found {invalid_values} invalid target values, dropping rows")
                df = df.dropna(subset=[target_column])
            
            # Remove negative and zero values (for demand forecasting)
            if target_column == 'Order_Demand':
                invalid_demand = (df[target_column] <= 0).sum()
                if invalid_demand > 0:
                    self.logger.warning(f"Removing {invalid_demand} non-positive demand values")
                    df = df[df[target_column] > 0]
            
            # Set date as index and sort
            df.set_index(date_column, inplace=True)
            df = df.sort_index()
            
            # Apply log transformation for demand
            if target_column == 'Order_Demand':
                df['Order_Demand_Log'] = np.log1p(df[target_column])
            
            self.logger.info("Data preprocessing completed",
                           final_rows=len(df),
                           date_range=f"{df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise DataError(f"Data preprocessing failed: {str(e)}", cause=e)
    
    def detect_missing_dates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect missing dates in time series"""
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        missing_dates = date_range.difference(df.index)
        
        return {
            "total_expected_dates": len(date_range),
            "actual_dates": len(df.index),
            "missing_dates_count": len(missing_dates),
            "missing_percentage": len(missing_dates) / len(date_range) * 100,
            "missing_dates": missing_dates.tolist()[:10]  # First 10 for display
        }


class StatisticalAnalyzer(BaseAnalyzer):
    """Professional statistical analysis with advanced methods"""
    
    @timer
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        with performance_monitor("statistical_analysis") as perf:
            results = {
                "descriptive_stats": self._descriptive_analysis(df),
                "distribution_analysis": self._distribution_analysis(df),
                "outlier_analysis": self._outlier_analysis(df),
                "correlation_analysis": self._correlation_analysis(df)
            }
            
            self.logger.info("Statistical analysis completed",
                           processing_time=perf.get('duration_seconds', 0))
            
            return results
    
    def _descriptive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive descriptive statistics"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        return {
            "basic_stats": numeric_df.describe().to_dict(),
            "skewness": numeric_df.skew().to_dict(),
            "kurtosis": numeric_df.kurtosis().to_dict(),
            "variance": numeric_df.var().to_dict()
        }
    
    def _distribution_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distributions"""
        results = {}
        
        if 'Order_Demand_Log' in df.columns:
            # Plot distribution
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            sns.histplot(df['Order_Demand'], kde=True, bins=30)
            plt.title('Original Order Demand Distribution')
            plt.xlabel('Order Demand')
            
            plt.subplot(1, 2, 2)
            sns.histplot(df['Order_Demand_Log'], kde=True, bins=30)
            plt.title('Log-Transformed Order Demand Distribution')
            plt.xlabel('Log(Order Demand)')
            
            plt.tight_layout()
            self._save_plot('demand_distribution.png')
            plt.show()
            
            results['normality_improved'] = (
                abs(df['Order_Demand_Log'].skew()) < abs(df['Order_Demand'].skew())
            )
        
        return results
    
    def _outlier_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive outlier detection"""
        results = {}
        
        if 'Order_Demand_Log' in df.columns:
            # IQR method
            Q1 = df['Order_Demand_Log'].quantile(0.25)
            Q3 = df['Order_Demand_Log'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df['Order_Demand_Log'] < lower_bound) | 
                         (df['Order_Demand_Log'] > upper_bound)]
            
            results['outlier_count'] = len(outliers)
            results['outlier_percentage'] = len(outliers) / len(df) * 100
            results['outlier_bounds'] = {'lower': lower_bound, 'upper': upper_bound}
            
            # Visualize outliers
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            sns.boxplot(data=df, y='Order_Demand_Log')
            plt.title('Outlier Detection: Log-Transformed Order Demand')
            
            plt.subplot(1, 2, 2)
            df['Order_Demand_Log'].plot(title='Time Series with Outliers Highlighted')
            outliers['Order_Demand_Log'].plot(style='ro', markersize=3)
            plt.xlabel('Date')
            plt.ylabel('Log(Order Demand)')
            plt.legend(['Normal Data', 'Outliers'])
            
            plt.tight_layout()
            self._save_plot('outlier_analysis.png')
            plt.show()
        
        return results
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive correlation analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5)
        plt.title('Correlation Heatmap')
        self._save_plot('correlation_heatmap.png')
        plt.show()
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "strong_correlations": self._find_strong_correlations(correlation_matrix)
        }
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, 
                                threshold: float = 0.7) -> List[Dict]:
        """Find strong correlations between variables"""
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corrs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return strong_corrs


class TimeSeriesAnalyzer(BaseAnalyzer):
    """Professional time series analysis"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not STATSMODELS_AVAILABLE:
            self.logger.warning("statsmodels not available, some features will be disabled")
    
    @timer
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive time series analysis"""
        with performance_monitor("time_series_analysis") as perf:
            results = {
                "stationarity_test": self._test_stationarity(df),
                "seasonality_analysis": self._analyze_seasonality(df),
                "trend_analysis": self._analyze_trends(df),
                "autocorrelation_analysis": self._analyze_autocorrelation(df)
            }
            
            if STATSMODELS_AVAILABLE:
                results["decomposition"] = self._decompose_time_series(df)
            
            self.logger.info("Time series analysis completed",
                           processing_time=perf.get('duration_seconds', 0))
            
            return results
    
    def _test_stationarity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test for stationarity using ADF test"""
        if not STATSMODELS_AVAILABLE or 'Order_Demand_Log' not in df.columns:
            return {"error": "Required libraries or columns not available"}
        
        try:
            result = adfuller(df['Order_Demand_Log'].dropna(), maxlag=10, autolag=None)
            
            stationarity_result = {
                "adf_statistic": result[0],
                "p_value": result[1],
                "critical_values": result[4],
                "is_stationary": result[1] <= 0.05
            }
            
            self.logger.info("Stationarity test completed",
                           is_stationary=stationarity_result["is_stationary"],
                           p_value=stationarity_result["p_value"])
            
            return stationarity_result
            
        except Exception as e:
            self.logger.error(f"Stationarity test failed: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns"""
        if 'Order_Demand' not in df.columns:
            return {"error": "Order_Demand column not found"}
        
        # Monthly seasonality
        monthly_data = df.resample('M')['Order_Demand'].sum()
        
        plt.figure(figsize=(15, 10))
        
        # Time series plot
        plt.subplot(2, 2, 1)
        df['Order_Demand'].plot(title='Order Demand Over Time')
        plt.xlabel('Date')
        plt.ylabel('Order Demand')
        
        # Monthly trends
        plt.subplot(2, 2, 2)
        monthly_data.plot(title='Monthly Order Demand')
        plt.xlabel('Month')
        plt.ylabel('Total Order Demand')
        
        # Seasonal patterns
        plt.subplot(2, 2, 3)
        df.groupby(df.index.month)['Order_Demand'].mean().plot(kind='bar',
                                                               title='Average Demand by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Demand')
        
        plt.subplot(2, 2, 4)
        df.groupby(df.index.dayofweek)['Order_Demand'].mean().plot(kind='bar',
                                                                   title='Average Demand by Day of Week')
        plt.xlabel('Day of Week (0=Monday)')
        plt.ylabel('Average Demand')
        
        plt.tight_layout()
        self._save_plot('seasonality_analysis.png')
        plt.show()
        
        return {
            "monthly_seasonality": df.groupby(df.index.month)['Order_Demand'].mean().to_dict(),
            "weekly_seasonality": df.groupby(df.index.dayofweek)['Order_Demand'].mean().to_dict()
        }
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze long-term trends"""
        if 'Order_Demand' not in df.columns:
            return {"error": "Order_Demand column not found"}
        
        # Calculate rolling averages
        df_trends = df.copy()
        df_trends['Rolling_7d'] = df_trends['Order_Demand'].rolling(window=7).mean()
        df_trends['Rolling_30d'] = df_trends['Order_Demand'].rolling(window=30).mean()
        df_trends['Rolling_90d'] = df_trends['Order_Demand'].rolling(window=90).mean()
        
        # Plot trends
        plt.figure(figsize=(15, 6))
        plt.plot(df_trends.index, df_trends['Order_Demand'], alpha=0.3, label='Daily Demand')
        plt.plot(df_trends.index, df_trends['Rolling_7d'], label='7-day MA')
        plt.plot(df_trends.index, df_trends['Rolling_30d'], label='30-day MA')
        plt.plot(df_trends.index, df_trends['Rolling_90d'], label='90-day MA')
        plt.title('Order Demand Trends with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Order Demand')
        plt.legend()
        self._save_plot('trend_analysis.png')
        plt.show()
        
        return {
            "overall_trend": "increasing" if df_trends['Rolling_90d'].iloc[-1] > df_trends['Rolling_90d'].iloc[90] else "decreasing",
            "volatility": df['Order_Demand'].std(),
            "trend_strength": abs(df_trends['Rolling_90d'].iloc[-1] - df_trends['Rolling_90d'].iloc[90]) / df['Order_Demand'].mean()
        }
    
    def _decompose_time_series(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components"""
        if not STATSMODELS_AVAILABLE or 'Order_Demand' not in df.columns:
            return {"error": "Required libraries or columns not available"}
        
        try:
            # Perform decomposition
            decomposition = seasonal_decompose(df['Order_Demand'], model='additive', period=30)
            
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            decomposition.observed.plot(ax=axes[0], title='Original Time Series')
            decomposition.trend.plot(ax=axes[1], title='Trend Component')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component')
            decomposition.resid.plot(ax=axes[3], title='Residual Component')
            
            plt.tight_layout()
            self._save_plot('time_series_decomposition.png')
            plt.show()
            
            return {
                "trend_strength": decomposition.trend.var() / df['Order_Demand'].var(),
                "seasonal_strength": decomposition.seasonal.var() / df['Order_Demand'].var(),
                "residual_variance": decomposition.resid.var()
            }
            
        except Exception as e:
            self.logger.error(f"Time series decomposition failed: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_autocorrelation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze autocorrelation patterns"""
        if 'Order_Demand' not in df.columns:
            return {"error": "Order_Demand column not found"}
        
        try:
            plt.figure(figsize=(12, 6))
            autocorrelation_plot(df['Order_Demand'])
            plt.title('Autocorrelation of Order Demand')
            self._save_plot('autocorrelation_plot.png')
            plt.show()
            
            return {"autocorrelation_analyzed": True}
            
        except Exception as e:
            self.logger.error(f"Autocorrelation analysis failed: {str(e)}")
            return {"error": str(e)}


class FeatureEngineer:
    """Professional feature engineering for time series"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.FeatureEngineer")
    
    @timer
    @log_method()
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for time series analysis"""
        df = df.copy()
        
        try:
            # Date-based features
            df['Day_of_Week'] = df.index.dayofweek
            df['Month'] = df.index.month
            df['Year'] = df.index.year
            df['Quarter'] = df.index.quarter
            df['Is_Weekend'] = (df.index.dayofweek >= 5).astype(int)
            df['Is_Month_Start'] = df.index.is_month_start.astype(int)
            df['Is_Month_End'] = df.index.is_month_end.astype(int)
            
            # Lag features
            if 'Order_Demand' in df.columns:
                for lag in [1, 7, 14, 30]:
                    df[f'Lag_{lag}'] = df['Order_Demand'].shift(lag)
                
                # Rolling statistics
                for window in [7, 14, 30]:
                    df[f'Rolling_Mean_{window}'] = df['Order_Demand'].rolling(window=window).mean()
                    df[f'Rolling_Std_{window}'] = df['Order_Demand'].rolling(window=window).std()
                    df[f'Rolling_Min_{window}'] = df['Order_Demand'].rolling(window=window).min()
                    df[f'Rolling_Max_{window}'] = df['Order_Demand'].rolling(window=window).max()
                
                # Difference features
                df['Demand_Diff_1'] = df['Order_Demand'].diff(1)
                df['Demand_Diff_7'] = df['Order_Demand'].diff(7)
            
            self.logger.info("Feature engineering completed",
                           original_features=len([col for col in df.columns if not col.startswith(('Lag_', 'Rolling_', 'Demand_'))]),
                           new_features=len([col for col in df.columns if col.startswith(('Lag_', 'Rolling_', 'Demand_'))]))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise DataError(f"Feature engineering failed: {str(e)}", cause=e)


class EDAAnalyzer:
    """Main EDA analyzer that orchestrates all analysis components"""
    
    def __init__(self, save_plots: bool = True, plot_dir: Optional[Path] = None):
        self.save_plots = save_plots
        self.plot_dir = plot_dir or Path("plots")
        self.plot_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.statistical_analyzer = StatisticalAnalyzer(save_plots, plot_dir)
        self.time_series_analyzer = TimeSeriesAnalyzer(save_plots, plot_dir)
        self.feature_engineer = FeatureEngineer()
        
        self.logger = get_logger(f"{__name__}.EDAAnalyzer")
    
    @timer
    def run_complete_eda(self, filepath: str, 
                        date_column: str = 'Date',
                        target_column: str = 'Order_Demand') -> EDAResult:
        """
        Run complete EDA analysis
        
        Args:
            filepath: Path to the data file
            date_column: Name of the date column
            target_column: Name of the target column
            
        Returns:
            EDAResult object with all analysis results
        """
        with performance_monitor("complete_eda") as perf:
            try:
                # Load data
                self.logger.info("Starting comprehensive EDA analysis")
                df = self.data_loader.load_data(filepath)
                dataset_info = self.data_loader.get_dataset_info(df)
                
                # Preprocess data
                df = self.preprocessor.preprocess_time_series_data(df, date_column, target_column)
                missing_data_analysis = self.preprocessor.detect_missing_dates(df)
                
                # Feature engineering
                df = self.feature_engineer.create_features(df)
                
                # Statistical analysis
                statistical_results = self.statistical_analyzer.analyze(df)
                
                # Time series analysis
                time_series_results = self.time_series_analyzer.analyze(df)
                
                # Create comprehensive result
                result = EDAResult(
                    dataset_info=dataset_info,
                    statistical_summary=statistical_results.get('descriptive_stats', {}),
                    missing_data_analysis=missing_data_analysis,
                    outlier_analysis=statistical_results.get('outlier_analysis', {}),
                    correlation_analysis=statistical_results.get('correlation_analysis', {}),
                    time_series_analysis=time_series_results
                )
                
                self.logger.info("EDA analysis completed successfully",
                               total_time=perf.get('duration_seconds', 0),
                               plots_saved=self.save_plots)
                
                return result
                
            except Exception as e:
                self.logger.error(f"EDA analysis failed: {str(e)}")
                raise DataError(f"EDA analysis failed: {str(e)}", cause=e)


class VisualizationManager:
    """Professional visualization management"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        self.style = style
        plt.style.use(style)
        self.logger = get_logger(f"{__name__}.VisualizationManager")
    
    def create_summary_dashboard(self, df: pd.DataFrame, save_path: Optional[Path] = None):
        """Create a comprehensive dashboard of visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Time series plot
        df['Order_Demand'].plot(ax=axes[0,0], title='Order Demand Over Time')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Order Demand')
        
        # Distribution plot
        sns.histplot(df['Order_Demand_Log'], kde=True, ax=axes[0,1])
        axes[0,1].set_title('Log-Transformed Demand Distribution')
        
        # Seasonal plot
        monthly_avg = df.groupby(df.index.month)['Order_Demand'].mean()
        monthly_avg.plot(kind='bar', ax=axes[0,2], title='Average Demand by Month')
        axes[0,2].set_xlabel('Month')
        
        # Outlier detection
        sns.boxplot(data=df, y='Order_Demand_Log', ax=axes[1,0])
        axes[1,0].set_title('Outlier Detection')
        
        # Correlation heatmap
        numeric_df = df.select_dtypes(include=[np.number]).iloc[:, :8]  # First 8 numeric columns
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
        axes[1,1].set_title('Correlation Heatmap')
        
        # Rolling averages
        df['Order_Demand'].rolling(30).mean().plot(ax=axes[1,2], title='30-Day Rolling Average')
        axes[1,2].set_xlabel('Date')
        axes[1,2].set_ylabel('30-Day Average')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dashboard saved to {save_path}")
        
        plt.show()


# Utility function for running EDA from the command line or main.py
def run_eda_analysis(filepath: str) -> EDAResult:
    """
    Convenience function to run EDA analysis
    
    Args:
        filepath: Path to the data file
        
    Returns:
        EDAResult object with analysis results
    """
    analyzer = EDAAnalyzer(save_plots=True)
    return analyzer.run_complete_eda(filepath)


# Export main classes
__all__ = [
    'EDAAnalyzer',
    'TimeSeriesAnalyzer',
    'StatisticalAnalyzer',
    'VisualizationManager',
    'EDAResult',
    'run_eda_analysis'
]