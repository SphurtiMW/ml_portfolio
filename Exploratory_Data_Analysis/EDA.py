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
    df = pd.read_csv(filepath)
    print("Dataset Loaded Successfully!")
    print("\nFirst 5 rows of the dataset:\n", df.head())
    print("\nDataset Information:\n")
    print(df.info())
    print("\nMissing Values:\n", df.isnull().sum())
    return df

# Preprocessing & Cleaning
def preprocess_data(df):
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

# Handle Outliers Using Log Transformation
def apply_log_transformation(df):
    df['Order_Demand_Log'] = np.log1p(df['Order_Demand'])  # log(1 + x) to avoid log(0)
    print("\n Log Transformation Applied!")
    return df

# Check for Missing Dates
def check_missing_dates(df):
    missing_dates = pd.date_range(start=df.index.min(), end=df.index.max()).difference(df.index)
    print(f"\nMissing Dates Count: {len(missing_dates)}")

# Basic Statistical Analysis
def descriptive_analysis(df):
    print("\n Basic Statistical Summary:\n")
    print(df.describe())

# Plot Demand Distribution
def plot_demand_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Order_Demand_Log'], kde=True, bins=30)
    plt.title('Log-Transformed Order Demand Distribution')
    plt.xlabel('Log(Order Demand)')
    plt.ylabel('Frequency')
    plt.show()

# Time Series Visualization
def plot_time_series(df):
    plt.figure(figsize=(12, 6))
    df['Order_Demand'].plot(title='Order Demand Over Time')
    plt.xlabel('Date')
    plt.ylabel('Order Demand')
    plt.show()

# Monthly Trends
def plot_monthly_trends(df):
    df.resample('M').sum()['Order_Demand'].plot(figsize=(12, 6), title='Monthly Order Demand')
    plt.xlabel('Month')
    plt.ylabel('Total Order Demand')
    plt.show()

# Feature Engineering
def create_features(df):
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

# Perform ADF Test
def test_stationarity(df):
    print("\n Performing Augmented Dickey-Fuller Test (ADF)...")
    result = adfuller(df['Order_Demand_Log'].dropna(), maxlag=10, autolag=None)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

    if result[1] > 0.05:
        print("Data is NOT stationary. Differencing may be required.")
    else:
        print("Data is stationary.")

# Time Series Decomposition
def decompose_time_series(df):
    decomposition = seasonal_decompose(df['Order_Demand'], model='additive', period=30)
    decomposition.plot()
    plt.show()

# Correlation Analysis (Fix: Drop Non-Numeric Columns)
def plot_correlation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(8, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Outlier Detection
def detect_outliers(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(df['Order_Demand_Log'])
    plt.title('Outlier Detection: Log-Transformed Order Demand')
    plt.show()

# Autocorrelation Check
def plot_autocorrelation(df):
    plt.figure(figsize=(10, 5))
    autocorrelation_plot(df['Order_Demand'])
    plt.title('Autocorrelation of Order Demand')
    plt.show()

# Main Function
def main():
    filepath = r"C:\Users\sphur\Downloads\Historical Product Demand.csv" 

    df = load_data(filepath)
    df = preprocess_data(df)
    df = apply_log_transformation(df)  
    check_missing_dates(df)
    descriptive_analysis(df)
    plot_demand_distribution(df)
    plot_time_series(df)
    plot_monthly_trends(df)
    df = create_features(df)
    test_stationarity(df)
    decompose_time_series(df)
    plot_correlation(df)
    detect_outliers(df)
    plot_autocorrelation(df)

# Run EDA
if __name__ == "__main__":
    main()
