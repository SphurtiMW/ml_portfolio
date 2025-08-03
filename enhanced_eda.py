import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
import warnings
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data(filepath):
    """Load dataset and perform initial exploration"""
    print("🔍 Loading and exploring dataset...")
    df = pd.read_csv(filepath)
    
    print("\n📊 DATASET OVERVIEW")
    print("=" * 50)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    print("\n📋 Dataset Info:")
    print(df.info())
    
    print("\n🔢 First 10 rows:")
    print(df.head(10))
    
    print("\n📊 Basic Statistics:")
    print(df.describe())
    
    print("\n❓ Missing Values:")
    missing_vals = df.isnull().sum()
    print(missing_vals[missing_vals > 0])
    
    print("\n🏷️ Unique Values per Column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
    
    return df

def preprocess_data(df):
    """Clean and preprocess the dataset"""
    print("\n🔧 PREPROCESSING DATA")
    print("=" * 50)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle Order_Demand column
    df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce')
    
    # Remove negative values and zeros
    initial_rows = len(df)
    df = df[df['Order_Demand'] > 0]
    print(f"Removed {initial_rows - len(df)} rows with zero/negative demand")
    
    # Remove missing values
    df = df.dropna()
    print(f"Final dataset shape after cleaning: {df.shape}")
    
    return df

def analyze_categorical_features(df):
    """Analyze categorical features in the dataset"""
    print("\n🏷️ CATEGORICAL FEATURES ANALYSIS")
    print("=" * 50)
    
    categorical_cols = ['Product_Code', 'Warehouse', 'Product_Category']
    
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  - Unique values: {df[col].nunique()}")
        print(f"  - Top 5 values:")
        print(df[col].value_counts().head())
    
    # Visualize distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, col in enumerate(categorical_cols):
        top_values = df[col].value_counts().head(10)
        axes[i].bar(range(len(top_values)), top_values.values)
        axes[i].set_title(f'Top 10 {col}')
        axes[i].set_xlabel(f'{col}')
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def demand_distribution_analysis(df):
    """Analyze demand distribution patterns"""
    print("\n📈 DEMAND DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    print(f"Order Demand Statistics:")
    print(f"  - Mean: {df['Order_Demand'].mean():.2f}")
    print(f"  - Median: {df['Order_Demand'].median():.2f}")
    print(f"  - Std: {df['Order_Demand'].std():.2f}")
    print(f"  - Min: {df['Order_Demand'].min()}")
    print(f"  - Max: {df['Order_Demand'].max()}")
    print(f"  - Skewness: {df['Order_Demand'].skew():.2f}")
    print(f"  - Kurtosis: {df['Order_Demand'].kurtosis():.2f}")
    
    # Create log transformation
    df['Order_Demand_Log'] = np.log1p(df['Order_Demand'])
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original distribution
    axes[0,0].hist(df['Order_Demand'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Original Order Demand Distribution')
    axes[0,0].set_xlabel('Order Demand')
    axes[0,0].set_ylabel('Frequency')
    
    # Log-transformed distribution
    axes[0,1].hist(df['Order_Demand_Log'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,1].set_title('Log-Transformed Order Demand Distribution')
    axes[0,1].set_xlabel('Log(Order Demand)')
    axes[0,1].set_ylabel('Frequency')
    
    # Box plot for outlier detection
    axes[1,0].boxplot(df['Order_Demand'])
    axes[1,0].set_title('Order Demand Box Plot')
    axes[1,0].set_ylabel('Order Demand')
    
    # Q-Q plot for normality check
    from scipy import stats
    stats.probplot(df['Order_Demand_Log'], dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot (Log-Transformed)')
    
    plt.tight_layout()
    plt.show()
    
    return df

def temporal_analysis(df):
    """Analyze temporal patterns in demand"""
    print("\n⏰ TEMPORAL ANALYSIS")
    print("=" * 50)
    
    # Create time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayName'] = df['Date'].dt.day_name()
    df['MonthName'] = df['Date'].dt.month_name()
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Years covered: {sorted(df['Year'].unique())}")
    
    # Aggregate by different time periods
    daily_demand = df.groupby('Date')['Order_Demand'].sum().reset_index()
    monthly_demand = df.groupby(['Year', 'Month'])['Order_Demand'].sum().reset_index()
    yearly_demand = df.groupby('Year')['Order_Demand'].sum()
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Daily demand over time
    axes[0,0].plot(daily_demand['Date'], daily_demand['Order_Demand'])
    axes[0,0].set_title('Daily Total Demand Over Time')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Total Order Demand')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Monthly patterns
    monthly_avg = df.groupby('Month')['Order_Demand'].mean()
    axes[0,1].bar(monthly_avg.index, monthly_avg.values)
    axes[0,1].set_title('Average Demand by Month')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Average Order Demand')
    
    # Day of week patterns
    dow_avg = df.groupby('DayName')['Order_Demand'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_avg = dow_avg.reindex(day_order)
    axes[1,0].bar(dow_avg.index, dow_avg.values)
    axes[1,0].set_title('Average Demand by Day of Week')
    axes[1,0].set_xlabel('Day of Week')
    axes[1,0].set_ylabel('Average Order Demand')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Yearly trends
    axes[1,1].bar(yearly_demand.index, yearly_demand.values)
    axes[1,1].set_title('Total Demand by Year')
    axes[1,1].set_xlabel('Year')
    axes[1,1].set_ylabel('Total Order Demand')
    
    plt.tight_layout()
    plt.show()
    
    return df

def product_warehouse_analysis(df):
    """Analyze demand patterns by product and warehouse"""
    print("\n🏭 PRODUCT & WAREHOUSE ANALYSIS")
    print("=" * 50)
    
    # Top products by total demand
    top_products = df.groupby('Product_Code')['Order_Demand'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
    print("\nTop 10 Products by Total Demand:")
    print(top_products.head(10))
    
    # Top warehouses by total demand
    top_warehouses = df.groupby('Warehouse')['Order_Demand'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
    print("\nWarehouse Performance:")
    print(top_warehouses)
    
    # Top categories by total demand
    top_categories = df.groupby('Product_Category')['Order_Demand'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
    print("\nTop 10 Categories by Total Demand:")
    print(top_categories.head(10))
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Top 15 products
    top_15_products = top_products.head(15)
    axes[0,0].barh(range(len(top_15_products)), top_15_products['sum'])
    axes[0,0].set_yticks(range(len(top_15_products)))
    axes[0,0].set_yticklabels(top_15_products.index)
    axes[0,0].set_title('Top 15 Products by Total Demand')
    axes[0,0].set_xlabel('Total Demand')
    
    # Warehouse comparison
    axes[0,1].bar(top_warehouses.index, top_warehouses['sum'])
    axes[0,1].set_title('Total Demand by Warehouse')
    axes[0,1].set_xlabel('Warehouse')
    axes[0,1].set_ylabel('Total Demand')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Top 15 categories
    top_15_categories = top_categories.head(15)
    axes[1,0].barh(range(len(top_15_categories)), top_15_categories['sum'])
    axes[1,0].set_yticks(range(len(top_15_categories)))
    axes[1,0].set_yticklabels(top_15_categories.index)
    axes[1,0].set_title('Top 15 Categories by Total Demand')
    axes[1,0].set_xlabel('Total Demand')
    
    # Demand distribution by warehouse
    df.boxplot(column='Order_Demand', by='Warehouse', ax=axes[1,1])
    axes[1,1].set_title('Demand Distribution by Warehouse')
    axes[1,1].set_xlabel('Warehouse')
    axes[1,1].set_ylabel('Order Demand')
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(df):
    """Perform correlation analysis on numerical features"""
    print("\n🔗 CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Select numerical columns
    numerical_cols = ['Order_Demand', 'Order_Demand_Log', 'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'IsWeekend']
    correlation_matrix = df[numerical_cols].corr()
    
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Visualize correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.show()

def time_series_analysis(df):
    """Perform time series specific analysis"""
    print("\n📊 TIME SERIES ANALYSIS")
    print("=" * 50)
    
    # Aggregate to daily level for time series analysis
    daily_ts = df.groupby('Date')['Order_Demand'].sum().sort_index()
    
    # Stationarity test
    print("Performing Augmented Dickey-Fuller Test...")
    adf_result = adfuller(daily_ts.dropna())
    print(f"ADF Statistic: {adf_result[0]:.6f}")
    print(f"p-value: {adf_result[1]:.6f}")
    print(f"Critical Values: {adf_result[4]}")
    
    if adf_result[1] <= 0.05:
        print("✅ Series is stationary")
    else:
        print("❌ Series is non-stationary")
    
    # Decomposition (if enough data points)
    if len(daily_ts) > 365:
        print("\nPerforming seasonal decomposition...")
        decomposition = seasonal_decompose(daily_ts, model='additive', period=30)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        decomposition.observed.plot(ax=axes[0], title='Original Time Series')
        decomposition.trend.plot(ax=axes[1], title='Trend Component')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component')
        decomposition.resid.plot(ax=axes[3], title='Residual Component')
        
        plt.tight_layout()
        plt.show()
    
    # Autocorrelation
    plt.figure(figsize=(12, 6))
    autocorrelation_plot(daily_ts)
    plt.title('Autocorrelation Plot of Daily Demand')
    plt.show()

def generate_insights_report(df):
    """Generate comprehensive insights report"""
    print("\n📝 COMPREHENSIVE INSIGHTS REPORT")
    print("=" * 80)
    
    # Data Overview
    print("🔍 DATA OVERVIEW:")
    print(f"   • Total records: {len(df):,}")
    print(f"   • Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   • Time span: {(df['Date'].max() - df['Date'].min()).days} days")
    print(f"   • Unique products: {df['Product_Code'].nunique():,}")
    print(f"   • Unique warehouses: {df['Warehouse'].nunique()}")
    print(f"   • Unique categories: {df['Product_Category'].nunique()}")
    
    # Demand Characteristics
    print(f"\n📊 DEMAND CHARACTERISTICS:")
    print(f"   • Total demand volume: {df['Order_Demand'].sum():,}")
    print(f"   • Average daily demand: {df['Order_Demand'].mean():.2f}")
    print(f"   • Demand volatility (CV): {(df['Order_Demand'].std() / df['Order_Demand'].mean()):.2f}")
    print(f"   • Peak demand: {df['Order_Demand'].max():,}")
    print(f"   • Most common demand value: {df['Order_Demand'].mode().iloc[0]:,}")
    
    # Temporal Patterns
    print(f"\n⏰ TEMPORAL PATTERNS:")
    yearly_growth = df.groupby('Year')['Order_Demand'].sum()
    if len(yearly_growth) > 1:
        growth_rate = ((yearly_growth.iloc[-1] / yearly_growth.iloc[0]) ** (1/(len(yearly_growth)-1)) - 1) * 100
        print(f"   • Annual growth rate: {growth_rate:.2f}%")
    
    peak_month = df.groupby('MonthName')['Order_Demand'].mean().idxmax()
    peak_day = df.groupby('DayName')['Order_Demand'].mean().idxmax()
    print(f"   • Peak demand month: {peak_month}")
    print(f"   • Peak demand day: {peak_day}")
    
    weekend_vs_weekday = df.groupby('IsWeekend')['Order_Demand'].mean()
    print(f"   • Weekend vs Weekday demand ratio: {weekend_vs_weekday[1]/weekend_vs_weekday[0]:.2f}")
    
    # Business Insights
    print(f"\n🏢 BUSINESS INSIGHTS:")
    top_product = df.groupby('Product_Code')['Order_Demand'].sum().idxmax()
    top_warehouse = df.groupby('Warehouse')['Order_Demand'].sum().idxmax()
    top_category = df.groupby('Product_Category')['Order_Demand'].sum().idxmax()
    
    print(f"   • Top performing product: {top_product}")
    print(f"   • Top performing warehouse: {top_warehouse}")
    print(f"   • Top performing category: {top_category}")
    
    # Data Quality Assessment
    print(f"\n✅ DATA QUALITY ASSESSMENT:")
    print(f"   • Missing values: {df.isnull().sum().sum()}")
    print(f"   • Duplicate records: {df.duplicated().sum()}")
    print(f"   • Zero demand records: {(df['Order_Demand'] == 0).sum()}")
    
    outlier_threshold = df['Order_Demand'].quantile(0.95)
    outliers = (df['Order_Demand'] > outlier_threshold).sum()
    print(f"   • Potential outliers (>95th percentile): {outliers}")
    
    print(f"\n🎯 RECOMMENDATIONS FOR FORECASTING:")
    print(f"   • Consider seasonal decomposition due to temporal patterns")
    print(f"   • Account for product-specific and warehouse-specific variations")
    print(f"   • Monitor outliers and high-demand events")
    print(f"   • Include external factors (holidays, promotions) if available")
    print(f"   • Consider ensemble methods for different product categories")

def main():
    """Main function to run comprehensive EDA"""
    filepath = "artifacts/Historical Product Demand.csv"
    
    try:
        # Load and explore data
        df = load_and_explore_data(filepath)
        
        # Preprocess data
        df = preprocess_data(df)
        
        # Perform various analyses
        analyze_categorical_features(df)
        df = demand_distribution_analysis(df)
        df = temporal_analysis(df)
        product_warehouse_analysis(df)
        correlation_analysis(df)
        time_series_analysis(df)
        
        # Generate final insights report
        generate_insights_report(df)
        
        print("\n✅ Exploratory Data Analysis completed successfully!")
        print("Check the visualizations and insights above for comprehensive understanding of your data.")
        
        return df
        
    except Exception as e:
        print(f"❌ Error during EDA: {str(e)}")
        raise

if __name__ == "__main__":
    df_analyzed = main()