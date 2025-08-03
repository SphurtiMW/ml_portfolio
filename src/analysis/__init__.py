"""
Analysis Module

This module provides professional EDA and analysis components:
- Exploratory Data Analysis (EDA) with professional visualizations
- Statistical analysis and hypothesis testing
- Time series analysis and decomposition
- Feature analysis and correlation studies
- Outlier detection and data quality assessment
"""

from .eda import (
    EDAAnalyzer,
    TimeSeriesAnalyzer,
    StatisticalAnalyzer,
    VisualizationManager
)

__all__ = [
    "EDAAnalyzer",
    "TimeSeriesAnalyzer", 
    "StatisticalAnalyzer",
    "VisualizationManager"
]