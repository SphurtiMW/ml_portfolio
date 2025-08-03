import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
import warnings
import os
import sys

warnings.filterwarnings("ignore")

# Load Dataset
def load_data(filepath):
    """Load dataset from CSV file with error handling"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print("Dataset Loaded Successfully!")
        print("\nFirst 5 rows of the dataset:\n", df.head())
        print("\nDataset Information:\n")
        print(df.info())
        print("\nMissing Values:\n", df.isnull().sum())
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Preprocessing & Cleaning
def preprocess_data(df):
    """Preprocess and clean the dataset"""
    if df is None:
        return None
    
    try:
        # Drop missing values
        df = df.dropna()

        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Convert 'Order_Demand' to numeric, handling parentheses if needed
        df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce')

        # Remove negative and zero values (if necessary)
        df = df[df['Order_Demand'] > 0]

        # Set Date as Index
        df.set_index('Date', inplace=True)
        
        # Ensure data is sorted by time
        df = df.sort_index()

        print("\n Data Preprocessing Completed!")
        print("\nMissing Values After Cleaning:\n", df.isnull().sum())

        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# Handle Outliers Using Log Transformation
def apply_log_transformation(df):
    """Apply log transformation to handle outliers"""
    if df is None:
        return None
    
    try:
        df['Order_Demand_Log'] = np.log1p(df['Order_Demand'])  # log(1 + x) to avoid log(0)
        print("\n Log Transformation Applied!")
        return df
    except Exception as e:
        print(f"Error during log transformation: {e}")
        return df

# Check for Missing Dates
def check_missing_dates(df):
    """Check for missing dates in the time series"""
    if df is None:
        return
    
    try:
        missing_dates = pd.date_range(start=df.index.min(), end=df.index.max()).difference(df.index)
        print(f"\nMissing Dates Count: {len(missing_dates)}")
    except Exception as e:
        print(f"Error checking missing dates: {e}")

# Basic Statistical Analysis
def descriptive_analysis(df):
    """Perform basic statistical analysis"""
    if df is None:
        return
    
    try:
        print("\n Basic Statistical Summary:\n")
        print(df.describe())
    except Exception as e:
        print(f"Error during descriptive analysis: {e}")

# Plot Demand Distribution
def plot_demand_distribution(df):
    """Plot distribution of log-transformed order demand"""
    if df is None or 'Order_Demand_Log' not in df.columns:
        return
    
    try:
        plt.figure(figsize=(8, 5))
        sns.histplot(df['Order_Demand_Log'], kde=True, bins=30)
        plt.title('Log-Transformed Order Demand Distribution')
        plt.xlabel('Log(Order Demand)')
        plt.ylabel('Frequency')
        plt.show()
    except Exception as e:
        print(f"Error plotting demand distribution: {e}")

# Time Series Visualization
def plot_time_series(df):
    """Plot time series of order demand"""
    if df is None:
        return
    
    try:
        plt.figure(figsize=(12, 6))
        df['Order_Demand'].plot(title='Order Demand Over Time')
        plt.xlabel('Date')
        plt.ylabel('Order Demand')
        plt.show()
    except Exception as e:
        print(f"Error plotting time series: {e}")

# Monthly Trends
def plot_monthly_trends(df):
    """Plot monthly aggregated trends"""
    if df is None:
        return
    
    try:
        df.resample('M').sum()['Order_Demand'].plot(figsize=(12, 6), title='Monthly Order Demand')
        plt.xlabel('Month')
        plt.ylabel('Total Order Demand')
        plt.show()
    except Exception as e:
        print(f"Error plotting monthly trends: {e}")

# Feature Engineering
def create_features(df):
    """Create additional features for analysis"""
    if df is None:
        return None
    
    try:
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)

        # Lag Features
        df['Lag_1'] = df['Order_Demand'].shift(1)
        df['Lag_7'] = df['Order_Demand'].shift(7)

        # Rolling Mean for Trend Analysis
        df['Rolling_Mean_7'] = df['Order_Demand'].rolling(window=7).mean()

        print("\n Feature Engineering Completed!\n")
        return df
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return df

# Perform ADF Test
def test_stationarity(df):
    """Perform Augmented Dickey-Fuller test for stationarity"""
    if df is None or 'Order_Demand_Log' not in df.columns:
        return
    
    try:
        print("\n Performing Augmented Dickey-Fuller Test (ADF)...")
        result = adfuller(df['Order_Demand_Log'].dropna(), maxlag=10, autolag=None)
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')

        if result[1] > 0.05:
            print("Data is NOT stationary. Differencing may be required.")
        else:
            print("Data is stationary.")
    except Exception as e:
        print(f"Error during stationarity test: {e}")

# Time Series Decomposition
def decompose_time_series(df):
    """Perform time series decomposition"""
    if df is None:
        return
    
    try:
        decomposition = seasonal_decompose(df['Order_Demand'], model='additive', period=30)
        decomposition.plot()
        plt.show()
    except Exception as e:
        print(f"Error during time series decomposition: {e}")

# Correlation Analysis (Fix: Drop Non-Numeric Columns)
def plot_correlation(df):
    """Plot correlation heatmap of numeric features"""
    if df is None:
        return
    
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()

        plt.figure(figsize=(8, 5))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
    except Exception as e:
        print(f"Error plotting correlation: {e}")

# Outlier Detection
def detect_outliers(df):
    """Detect outliers using boxplot"""
    if df is None or 'Order_Demand_Log' not in df.columns:
        return
    
    try:
        plt.figure(figsize=(8, 5))
        sns.boxplot(df['Order_Demand_Log'])
        plt.title('Outlier Detection: Log-Transformed Order Demand')
        plt.show()
    except Exception as e:
        print(f"Error detecting outliers: {e}")

# Autocorrelation Check
def plot_autocorrelation(df):
    """Plot autocorrelation of order demand"""
    if df is None:
        return
    
    try:
        plt.figure(figsize=(10, 5))
        autocorrelation_plot(df['Order_Demand'])
        plt.title('Autocorrelation of Order Demand')
        plt.show()
    except Exception as e:
        print(f"Error plotting autocorrelation: {e}")

def get_data_file_path():
    """Get the path to the data file, checking multiple possible locations"""
    possible_paths = [
        "Historical Product Demand.csv",  # Current directory
        "data/Historical Product Demand.csv",  # Data subdirectory
        "../data/Historical Product Demand.csv",  # Parent data directory
        os.path.expanduser("~/Downloads/Historical Product Demand.csv"),  # Downloads folder
        "Exploratory_Data_Analysis/Historical Product Demand.csv"  # EDA directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If no file found, ask user to provide path
    print("Data file not found in common locations.")
    print("Please provide the full path to your 'Historical Product Demand.csv' file:")
    user_path = input("File path: ").strip().strip('"').strip("'")
    
    if os.path.exists(user_path):
        return user_path
    else:
        print(f"File not found at: {user_path}")
        return None

# Main Function
def main():
    """Main function to run the complete EDA pipeline"""
    print("Starting Exploratory Data Analysis...")
    
    # Get data file path
    filepath = get_data_file_path()
    
    if filepath is None:
        print("Error: Could not locate the data file. Please ensure 'Historical Product Demand.csv' is available.")
        return
    
    print(f"Using data file: {filepath}")
    
    # Load and process data
    df = load_data(filepath)
    if df is None:
        return
    
    df = preprocess_data(df)
    if df is None:
        return
        
    df = apply_log_transformation(df)
    if df is None:
        return
    
    # Perform analysis
    check_missing_dates(df)
    descriptive_analysis(df)
    plot_demand_distribution(df)
    plot_time_series(df)
    plot_monthly_trends(df)
    
    df = create_features(df)
    if df is None:
        return
        
    test_stationarity(df)
    decompose_time_series(df)
    plot_correlation(df)
    detect_outliers(df)
    plot_autocorrelation(df)
    
    print("\nEDA completed successfully!")

# Run EDA
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEDA interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
