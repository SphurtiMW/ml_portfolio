#!/usr/bin/env python3
"""
Enhanced EDA for Time Series Demand Forecasting
This script provides comprehensive analysis for time series data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load and preprocess the time series data"""
    print("=== DATA LOADING AND PREPROCESSING ===")
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"Original dataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Convert to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle missing values
    print(f"\nMissing values before cleaning:")
    print(df.isnull().sum())
    
    # Drop missing values
    df = df.dropna()
    print(f"\nDataset shape after cleaning: {df.shape}")
    
    # Set date as index
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    
    return df

def basic_statistics(df):
    """Perform basic statistical analysis"""
    print("\n=== BASIC STATISTICAL ANALYSIS ===")
    
    print("Descriptive Statistics:")
    print(df['Order_Demand'].describe())
    
    print(f"\nData Range: {df.index.min()} to {df.index.max()}")
    print(f"Total Days: {len(df)}")
    print(f"Missing Days: {365 - len(df)}")
    
    # Check for outliers using IQR method
    Q1 = df['Order_Demand'].quantile(0.25)
    Q3 = df['Order_Demand'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['Order_Demand'] < lower_bound) | (df['Order_Demand'] > upper_bound)]
    print(f"\nOutliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    return df

def time_series_analysis(df):
    """Perform time series specific analysis"""
    print("\n=== TIME SERIES ANALYSIS ===")
    
    # Create time-based features
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Day_of_Month'] = df.index.day
    df['Week_of_Year'] = df.index.isocalendar().week
    
    # Weekly patterns
    print("\nWeekly Patterns (Average Demand by Day of Week):")
    weekly_avg = df.groupby('Day_of_Week')['Order_Demand'].mean()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, day in enumerate(days):
        print(f"{day}: {weekly_avg[i]:.1f}")
    
    # Monthly patterns
    print("\nMonthly Patterns (Average Demand by Month):")
    monthly_avg = df.groupby('Month')['Order_Demand'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i, month in enumerate(months):
        if i+1 in monthly_avg.index:
            print(f"{month}: {monthly_avg[i+1]:.1f}")
    
    return df

def trend_analysis(df):
    """Analyze trends in the data"""
    print("\n=== TREND ANALYSIS ===")
    
    # Calculate rolling statistics
    df['Rolling_Mean_7'] = df['Order_Demand'].rolling(window=7).mean()
    df['Rolling_Mean_30'] = df['Order_Demand'].rolling(window=30).mean()
    
    # Simple trend calculation
    x = np.arange(len(df))
    y = df['Order_Demand'].values
    trend_coef = np.polyfit(x, y, 1)
    trend_slope = trend_coef[0]
    
    print(f"Overall trend slope: {trend_slope:.2f}")
    if trend_slope > 0:
        print("Trend: Increasing demand over time")
    elif trend_slope < 0:
        print("Trend: Decreasing demand over time")
    else:
        print("Trend: No significant trend detected")
    
    return df

def seasonality_analysis(df):
    """Analyze seasonality patterns"""
    print("\n=== SEASONALITY ANALYSIS ===")
    
    # Weekly seasonality
    weekly_pattern = df.groupby('Day_of_Week')['Order_Demand'].mean()
    weekly_std = df.groupby('Day_of_Week')['Order_Demand'].std()
    
    print("Weekly Seasonality (Mean ± Std):")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, day in enumerate(days):
        print(f"{day}: {weekly_pattern[i]:.1f} ± {weekly_std[i]:.1f}")
    
    # Monthly seasonality
    monthly_pattern = df.groupby('Month')['Order_Demand'].mean()
    monthly_std = df.groupby('Month')['Order_Demand'].std()
    
    print("\nMonthly Seasonality (Mean ± Std):")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i, month in enumerate(months):
        if i+1 in monthly_pattern.index:
            print(f"{month}: {monthly_pattern[i+1]:.1f} ± {monthly_std[i+1]:.1f}")
    
    return df

def stationarity_check(df):
    """Check for stationarity using basic methods"""
    print("\n=== STATIONARITY CHECK ===")
    
    # Calculate rolling statistics
    rolling_mean = df['Order_Demand'].rolling(window=30).mean()
    rolling_std = df['Order_Demand'].rolling(window=30).std()
    
    # Check if rolling statistics are relatively constant
    mean_variance = rolling_mean.var()
    std_variance = rolling_std.var()
    
    print(f"Rolling mean variance: {mean_variance:.2f}")
    print(f"Rolling std variance: {std_variance:.2f}")
    
    if mean_variance < 100 and std_variance < 50:
        print("Data appears relatively stationary")
    else:
        print("Data appears non-stationary - differencing may be needed")
    
    return df

def autocorrelation_analysis(df):
    """Analyze autocorrelation patterns"""
    print("\n=== AUTOCORRELATION ANALYSIS ===")
    
    # Calculate autocorrelation for different lags
    lags = [1, 7, 14, 30]
    
    for lag in lags:
        if lag < len(df):
            autocorr = df['Order_Demand'].autocorr(lag=lag)
            print(f"Autocorrelation at lag {lag}: {autocorr:.3f}")
    
    return df

def generate_recommendations(df):
    """Generate recommendations for model training"""
    print("\n=== RECOMMENDATIONS FOR MODEL TRAINING ===")
    
    # Analyze data characteristics
    total_days = len(df)
    missing_days = 365 - total_days
    data_completeness = (total_days / 365) * 100
    
    print(f"Data Completeness: {data_completeness:.1f}% ({total_days}/365 days)")
    
    # Check for seasonality
    weekly_variance = df.groupby('Day_of_Week')['Order_Demand'].var().mean()
    monthly_variance = df.groupby('Month')['Order_Demand'].var().mean()
    
    print(f"Weekly pattern variance: {weekly_variance:.2f}")
    print(f"Monthly pattern variance: {monthly_variance:.2f}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if data_completeness < 90:
        print("⚠️  WARNING: Missing data detected. Consider imputation methods.")
    
    if weekly_variance > 100:
        print("✅ Strong weekly seasonality detected. Include day-of-week features.")
    
    if monthly_variance > 500:
        print("✅ Strong monthly seasonality detected. Include month features.")
    
    # Check data size for model training
    if total_days < 100:
        print("⚠️  Limited data for training. Consider simpler models.")
    elif total_days < 200:
        print("✅ Moderate data size. Standard time series models should work.")
    else:
        print("✅ Good data size for model training.")
    
    # Check for outliers
    Q1 = df['Order_Demand'].quantile(0.25)
    Q3 = df['Order_Demand'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['Order_Demand'] < Q1 - 1.5*IQR) | (df['Order_Demand'] > Q3 + 1.5*IQR)]
    
    if len(outliers) > 0:
        print(f"⚠️  {len(outliers)} outliers detected. Consider outlier handling.")
    
    print("\nSUGGESTED NEXT STEPS:")
    print("1. ✅ Proceed with model training")
    print("2. Include day-of-week and month features")
    print("3. Consider seasonal decomposition")
    print("4. Test both ARIMA and Prophet models")
    print("5. Use cross-validation with time series splits")

def main():
    """Main function to run enhanced EDA"""
    print("ENHANCED EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Load and preprocess data
    df = load_and_preprocess_data("Historical Product Demand.csv")
    
    if df is None or len(df) == 0:
        print("Error: No data to analyze")
        return
    
    # Perform analysis
    df = basic_statistics(df)
    df = time_series_analysis(df)
    df = trend_analysis(df)
    df = seasonality_analysis(df)
    df = stationarity_check(df)
    df = autocorrelation_analysis(df)
    generate_recommendations(df)
    
    print("\n" + "=" * 50)
    print("ENHANCED EDA COMPLETED!")

if __name__ == "__main__":
    main()