#!/usr/bin/env python3
"""
Sample data generator for testing the EDA script
Creates a sample 'Historical Product Demand.csv' file with realistic time series data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(filename="Historical Product Demand.csv", num_days=365):
    """Generate sample time series data for product demand"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # Generate base demand with trend and seasonality
    base_demand = 100
    trend = np.linspace(0, 50, num_days)  # Increasing trend
    
    # Weekly seasonality (higher demand on weekends)
    weekly_cycle = 20 * np.sin(2 * np.pi * np.arange(num_days) / 7)
    
    # Monthly seasonality
    monthly_cycle = 15 * np.sin(2 * np.pi * np.arange(num_days) / 30)
    
    # Random noise
    noise = np.random.normal(0, 10, num_days)
    
    # Combine all components
    demand = base_demand + trend + weekly_cycle + monthly_cycle + noise
    
    # Ensure all values are positive
    demand = np.maximum(demand, 1)
    
    # Add some outliers
    outlier_indices = np.random.choice(num_days, size=int(num_days * 0.05), replace=False)
    demand[outlier_indices] *= np.random.uniform(2, 4, len(outlier_indices))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Order_Demand': demand.round(0).astype(int)
    })
    
    # Add some missing values randomly (about 2%)
    missing_indices = np.random.choice(num_days, size=int(num_days * 0.02), replace=False)
    df.loc[missing_indices, 'Order_Demand'] = np.nan
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Sample data generated and saved to: {filename}")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Demand range: {df['Order_Demand'].min():.0f} to {df['Order_Demand'].max():.0f}")
    print(f"Missing values: {df['Order_Demand'].isnull().sum()}")
    
    return df

if __name__ == "__main__":
    # Generate sample data
    sample_df = generate_sample_data()
    print("\nFirst 10 rows of generated data:")
    print(sample_df.head(10))