# 🎯 Guided Exploratory Data Analysis Tutorial
## Supply Chain Demand Forecasting Dataset

### 📚 Learning Objectives
By the end of this tutorial, you'll be able to:
- Load and inspect large datasets efficiently
- Identify data quality issues and handle missing values
- Analyze temporal patterns in time series data
- Understand business dimensions and their impact
- Create meaningful visualizations for insights
- Generate actionable recommendations for forecasting models

---

## 🚀 Step 1: Initial Data Exploration
**Goal: Get familiar with your dataset structure**

### Your Tasks:
1. Load the dataset from `artifacts/Historical Product Demand.csv`
2. Display basic information about the dataset:
   - Shape (rows, columns)
   - Column names and data types
   - Memory usage
   - First few rows

### Key Questions to Answer:
- How big is the dataset?
- What columns do you have?
- What time period does the data cover?
- Are there any obvious data type issues?

### Code Template:
```python
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('artifacts/Historical Product Demand.csv')

# Your code here to explore:
# - df.shape
# - df.info()
# - df.head()
# - df.columns
```

### 💡 Hints:
- Pay attention to any warnings when loading the data
- Check if all columns are the expected data types
- Notice the date format - you'll need to convert it later

**Try this step first, then let me know what you find!**

---

## 🔍 Step 2: Data Quality Assessment
**Goal: Identify and understand data quality issues**

### Your Tasks:
1. Check for missing values in each column
2. Look for duplicate records
3. Examine the data types more carefully
4. Calculate the percentage of complete vs incomplete records

### Key Questions to Answer:
- How many missing values are in each column?
- What percentage of your data is actually usable?
- Are there any unexpected patterns in missing data?
- Do you have duplicate entries?

### Code Template:
```python
# Check missing values
missing_info = df.isnull().sum()
print("Missing values per column:")
print(missing_info)

# Calculate percentages
total_rows = len(df)
missing_percentage = (missing_info / total_rows) * 100
print("\nMissing value percentages:")
print(missing_percentage)

# Your code here for:
# - df.duplicated().sum()
# - Complete records analysis
```

**This step will reveal a major finding - see what you discover!**

---

## 📊 Step 3: Focus on Clean Data
**Goal: Work with the usable portion of your dataset**

### Your Tasks:
1. Create a clean dataset by removing rows with missing values
2. Convert the Date column to datetime format
3. Compare the original vs clean dataset sizes
4. Display basic statistics for the clean data

### Code Template:
```python
# Create clean dataset
df_clean = df.dropna()
print(f"Original dataset: {df.shape}")
print(f"Clean dataset: {df_clean.shape}")

# Convert date column
df_clean['Date'] = pd.to_datetime(df_clean['Date'])

# Your code here for:
# - df_clean.describe()
# - Date range analysis
# - Basic info about clean data
```

### 💡 Hints:
- You might get a warning about modifying a copy - that's okay for now
- Look at the dramatic difference in dataset size
- Check the date range to understand your time span

---

## 📈 Step 4: Demand Distribution Analysis
**Goal: Understand the characteristics of demand values**

### Your Tasks:
1. Calculate basic statistics for Order_Demand
2. Examine the distribution shape (skewness, kurtosis)
3. Find percentiles to understand the spread
4. Identify potential outliers

### Key Questions to Answer:
- Is the demand distribution normal or skewed?
- What's the typical demand range?
- How volatile is the demand?
- Are there extreme outliers?

### Code Template:
```python
demand = df_clean['Order_Demand']

# Basic statistics
print("Demand Statistics:")
print(f"Mean: {demand.mean():.2f}")
print(f"Median: {demand.median():.2f}")
print(f"Std Dev: {demand.std():.2f}")
print(f"Min: {demand.min()}")
print(f"Max: {demand.max()}")

# Distribution shape
print(f"\nDistribution Shape:")
print(f"Skewness: {demand.skew():.2f}")
print(f"Kurtosis: {demand.kurtosis():.2f}")

# Your code here for:
# - Percentiles: np.percentile(demand, [25, 50, 75, 90, 95, 99])
# - Coefficient of variation
```

### 💡 Key Insights to Look For:
- A skewness > 2 indicates highly skewed data
- High coefficient of variation means high volatility
- Compare mean vs median for outlier impact

---

## ⏰ Step 5: Temporal Pattern Analysis
**Goal: Discover time-based patterns in demand**

### Your Tasks:
1. Create time-based features (year, month, day of week)
2. Analyze yearly trends
3. Examine monthly seasonality
4. Investigate day-of-week patterns
5. Compare weekend vs weekday demand

### Code Template:
```python
# Create time features
df_clean['Year'] = df_clean['Date'].dt.year
df_clean['Month'] = df_clean['Date'].dt.month
df_clean['DayOfWeek'] = df_clean['Date'].dt.dayofweek
df_clean['IsWeekend'] = df_clean['DayOfWeek'].isin([5, 6]).astype(int)

# Yearly analysis
yearly_stats = df_clean.groupby('Year')['Order_Demand'].agg(['count', 'sum', 'mean'])
print("Yearly Statistics:")
print(yearly_stats)

# Your code here for:
# - Monthly patterns
# - Day of week analysis
# - Weekend vs weekday comparison
```

### 💡 Look For:
- Are there clear seasonal patterns?
- Which days of the week are busiest?
- Is there a weekend effect?

---

## 🏭 Step 6: Business Dimension Analysis
**Goal: Understand how business factors affect demand**

### Your Tasks:
1. Analyze warehouse performance
2. Identify top-performing products
3. Examine product category patterns
4. Calculate market share distributions

### Code Template:
```python
# Warehouse analysis
warehouse_stats = df_clean.groupby('Warehouse')['Order_Demand'].agg(['count', 'sum', 'mean'])
print("Warehouse Performance:")
print(warehouse_stats)

# Top products
top_products = df_clean.groupby('Product_Code')['Order_Demand'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products:")
print(top_products)

# Your code here for:
# - Product categories analysis
# - Market share calculations
# - Unique counts per dimension
```

---

## 📊 Step 7: Create Basic Visualizations
**Goal: Visualize patterns to confirm your findings**

### Your Tasks:
1. Create a histogram of demand distribution
2. Plot time series of daily total demand
3. Show monthly average demand patterns
4. Compare warehouse performance visually

### Code Template:
```python
import matplotlib.pyplot as plt

# Set up plotting
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Demand distribution
axes[0,0].hist(df_clean['Order_Demand'], bins=50, alpha=0.7)
axes[0,0].set_title('Demand Distribution')
axes[0,0].set_xlabel('Order Demand')
axes[0,0].set_ylabel('Frequency')

# Your code here for other plots:
# - Time series plot
# - Monthly patterns
# - Warehouse comparison
```

---

## 🎯 Step 8: Generate Insights and Recommendations
**Goal: Translate findings into actionable insights**

### Your Tasks:
1. Summarize key patterns discovered
2. Identify data quality issues
3. List challenges for forecasting
4. Provide recommendations for model improvement

### Questions to Consider:
- What are the biggest data quality challenges?
- Which patterns are most important for forecasting?
- What preprocessing steps are needed?
- How should you structure your forecasting approach?

---

## 🚀 Getting Started

**Start with Step 1** and work through each step at your own pace. Here's how I'll help:

1. **Try each step yourself first** - This is the best way to learn
2. **Ask specific questions** when you get stuck
3. **Share your findings** and I'll provide feedback
4. **Reference my complete solution** only when needed
5. **Compare approaches** after you've tried your own

### Ready to Begin?

Start with Step 1 - load your data and explore its basic structure. Share what you find, and I'll guide you to the next step!

Remember: There's no "wrong" way to explore data. The goal is to understand your dataset deeply so you can build better models.

**What questions do you have before starting? Or go ahead and try Step 1 - I'm here to help!**