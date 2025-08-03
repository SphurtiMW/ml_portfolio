#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_basic_plots():
    """Create basic visualizations for the supply chain demand data"""
    
    print("📊 Creating visualizations for Supply Chain Demand Analysis...")
    
    # Load and clean data
    df = pd.read_csv('artifacts/Historical Product Demand.csv', 
                     dtype={'Product_Code': str, 'Warehouse': str, 'Product_Category': str})
    df_clean = df.dropna()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Demand Distribution Histogram
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(df_clean['Order_Demand'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Order Demand Distribution')
    plt.xlabel('Order Demand')
    plt.ylabel('Frequency')
    plt.yscale('log')
    
    # 2. Log-transformed Distribution
    plt.subplot(2, 3, 2)
    log_demand = np.log1p(df_clean['Order_Demand'])
    plt.hist(log_demand, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Log-Transformed Demand Distribution')
    plt.xlabel('Log(Order Demand)')
    plt.ylabel('Frequency')
    
    # 3. Monthly Demand Pattern
    plt.subplot(2, 3, 3)
    monthly_demand = df_clean.groupby(df_clean['Date'].dt.month)['Order_Demand'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(range(1, 13), monthly_demand.values, color='coral')
    plt.title('Average Monthly Demand')
    plt.xlabel('Month')
    plt.ylabel('Average Demand')
    plt.xticks(range(1, 13), months, rotation=45)
    
    # 4. Day of Week Pattern
    plt.subplot(2, 3, 4)
    dow_demand = df_clean.groupby(df_clean['Date'].dt.dayofweek)['Order_Demand'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.bar(range(7), dow_demand.values, color='gold')
    plt.title('Average Demand by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Demand')
    plt.xticks(range(7), days)
    
    # 5. Warehouse Comparison
    plt.subplot(2, 3, 5)
    warehouse_demand = df_clean.groupby('Warehouse')['Order_Demand'].sum()
    plt.bar(warehouse_demand.index, warehouse_demand.values, color='lightcoral')
    plt.title('Total Demand by Warehouse')
    plt.xlabel('Warehouse')
    plt.ylabel('Total Demand')
    plt.xticks(rotation=45)
    
    # 6. Top Categories
    plt.subplot(2, 3, 6)
    top_categories = df_clean.groupby('Product_Category')['Order_Demand'].sum().head(10)
    plt.barh(range(len(top_categories)), top_categories.values, color='lightblue')
    plt.title('Top 10 Categories by Total Demand')
    plt.xlabel('Total Demand')
    plt.yticks(range(len(top_categories)), top_categories.index)
    
    plt.tight_layout()
    plt.savefig('demand_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Time Series Plot
    plt.figure(figsize=(15, 6))
    daily_demand = df_clean.groupby('Date')['Order_Demand'].sum().sort_index()
    plt.plot(daily_demand.index, daily_demand.values, color='blue', alpha=0.7)
    plt.title('Daily Total Demand Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Daily Demand')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('time_series_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations created and saved as:")
    print("  - demand_analysis_plots.png")
    print("  - time_series_plot.png")

if __name__ == "__main__":
    create_basic_plots()