import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
import warnings

warnings.filterwarnings("ignore")

# Load Dataset
def load_data(filepath):
    print("🔍 Loading dataset...")
    df = pd.read_csv(filepath, low_memory=False)  # Added low_memory=False to handle mixed types
    print("✅ Dataset Loaded Successfully!")
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Information:")
    print(df.info())
    print("\nMissing Values:")
    missing_vals = df.isnull().sum()
    print(missing_vals)
    return df

# Preprocessing & Cleaning
def preprocess_data(df):
    print("\n🔧 Starting data preprocessing...")
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    print(f"Original dataset shape: {df.shape}")
    
    # Drop missing values
    df = df.dropna()
    print(f"After dropping missing values: {df.shape}")

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Convert 'Order_Demand' to numeric, handling parentheses if needed
    df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce')

    # Remove negative and zero values (if necessary)
    df = df[df['Order_Demand'] > 0]
    print(f"After removing zero/negative values: {df.shape}")

    # Set Date as Index
    df.set_index('Date', inplace=True)
    
    # Ensure data is sorted by time
    df = df.sort_index()

    print("✅ Data Preprocessing Completed!")
    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    return df

# Handle Outliers Using Log Transformation
def apply_log_transformation(df):
    df = df.copy()  # Make a copy to avoid warnings
    df['Order_Demand_Log'] = np.log1p(df['Order_Demand'])  # log(1 + x) to avoid log(0)
    print("✅ Log Transformation Applied!")
    return df

# Check for Missing Dates
def check_missing_dates(df):
    missing_dates = pd.date_range(start=df.index.min(), end=df.index.max()).difference(df.index)
    print(f"\n📅 Missing Dates Count: {len(missing_dates)}")
    if len(missing_dates) > 0:
        print("This indicates gaps in your time series data")

# Basic Statistical Analysis
def descriptive_analysis(df):
    print("\n📊 Basic Statistical Summary:")
    print(df.describe())

# Plot Demand Distribution
def plot_demand_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Order_Demand_Log'], kde=True, bins=30)
    plt.title('Log-Transformed Order Demand Distribution')
    plt.xlabel('Log(Order Demand)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('demand_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

# Time Series Visualization
def plot_time_series(df):
    plt.figure(figsize=(12, 6))
    df['Order_Demand'].plot(title='Order Demand Over Time')
    plt.xlabel('Date')
    plt.ylabel('Order Demand')
    plt.tight_layout()
    plt.savefig('time_series.png', dpi=150, bbox_inches='tight')
    plt.show()

# Monthly Trends
def plot_monthly_trends(df):
    monthly_data = df.resample('M').sum()['Order_Demand']
    plt.figure(figsize=(12, 6))
    monthly_data.plot(title='Monthly Order Demand')
    plt.xlabel('Month')
    plt.ylabel('Total Order Demand')
    plt.tight_layout()
    plt.savefig('monthly_trends.png', dpi=150, bbox_inches='tight')
    plt.show()

# Feature Engineering
def create_features(df):
    df = df.copy()  # Make a copy to avoid warnings
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)

    # Lag Features
    df['Lag_1'] = df['Order_Demand'].shift(1)
    df['Lag_7'] = df['Order_Demand'].shift(7)

    # Rolling Mean for Trend Analysis
    df['Rolling_Mean_7'] = df['Order_Demand'].rolling(window=7).mean()

    print("✅ Feature Engineering Completed!")
    return df

# Perform ADF Test
def test_stationarity(df):
    print("\n📈 Performing Augmented Dickey-Fuller Test (ADF)...")
    try:
        result = adfuller(df['Order_Demand_Log'].dropna(), maxlag=10, autolag=None)
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')

        if result[1] > 0.05:
            print("❌ Data is NOT stationary. Differencing may be required.")
        else:
            print("✅ Data is stationary.")
    except Exception as e:
        print(f"Error in stationarity test: {e}")

# Time Series Decomposition
def decompose_time_series(df):
    try:
        print("\n📊 Performing time series decomposition...")
        # Resample to daily level for decomposition
        daily_data = df.resample('D').sum()['Order_Demand']
        daily_data = daily_data[daily_data > 0]  # Remove zeros
        
        if len(daily_data) > 60:  # Need enough data points
            decomposition = seasonal_decompose(daily_data, model='additive', period=30)
            decomposition.plot()
            plt.tight_layout()
            plt.savefig('decomposition.png', dpi=150, bbox_inches='tight')
            plt.show()
        else:
            print("Not enough data points for decomposition")
    except Exception as e:
        print(f"Error in decomposition: {e}")

# Correlation Analysis
def plot_correlation(df):
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error in correlation plot: {e}")

# Outlier Detection
def detect_outliers(df):
    try:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, y='Order_Demand_Log')
        plt.title('Outlier Detection: Log-Transformed Order Demand')
        plt.tight_layout()
        plt.savefig('outliers.png', dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error in outlier detection: {e}")

# Autocorrelation Check
def plot_autocorrelation(df):
    try:
        plt.figure(figsize=(10, 5))
        # Use daily aggregated data for autocorrelation
        daily_data = df.resample('D').sum()['Order_Demand']
        autocorrelation_plot(daily_data)
        plt.title('Autocorrelation of Daily Order Demand')
        plt.tight_layout()
        plt.savefig('autocorrelation.png', dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error in autocorrelation plot: {e}")

# Main Function
def main():
    print("🚀 Starting Comprehensive EDA...")
    
    # FIXED: Update file path for your environment
    filepath = "artifacts/Historical Product Demand.csv"  # Changed from Windows path
    
    try:
        df = load_data(filepath)
        df = preprocess_data(df)
        df = apply_log_transformation(df)  
        check_missing_dates(df)
        descriptive_analysis(df)
        
        print("\n📊 Creating visualizations...")
        plot_demand_distribution(df)
        plot_time_series(df)
        plot_monthly_trends(df)
        
        df = create_features(df)
        test_stationarity(df)
        decompose_time_series(df)
        plot_correlation(df)
        detect_outliers(df)
        plot_autocorrelation(df)
        
        print("\n✅ EDA completed successfully!")
        print("📁 Check the generated PNG files for visualizations")
        
    except FileNotFoundError:
        print("❌ File not found! Make sure 'artifacts/Historical Product Demand.csv' exists")
        print("Current directory contents:")
        import os
        print(os.listdir('.'))
    except Exception as e:
        print(f"❌ Error during EDA: {e}")

# Run EDA
if __name__ == "__main__":
    main()