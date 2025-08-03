# Exploratory Data Analysis Report
## Supply Chain Demand Forecasting Dataset

### 📊 Executive Summary

This comprehensive exploratory data analysis reveals key insights into the Historical Product Demand dataset used for LSTM-based supply chain forecasting. The dataset contains over 1 million records spanning from November 2011 to January 2013, covering 516 unique products across 3 warehouses and 22 product categories.

---

## 🔍 Dataset Overview

**Key Metrics:**
- **Total Records:** 1,048,575 (7,999 complete records after cleaning)
- **Time Period:** November 25, 2011 - January 7, 2013 (409 days)
- **Total Demand Volume:** 141,936,692 units
- **Memory Usage:** 136.82 MB

**Data Structure:**
```
Columns: ['Product_Code', 'Warehouse', 'Product_Category', 'Date', 'Order_Demand']
- Product_Code: 516 unique products
- Warehouse: 3 warehouses (Whse_C, Whse_J, Whse_S)
- Product_Category: 22 categories
- Date: 316 unique dates
- Order_Demand: 520 unique demand values
```

---

## 📈 Demand Distribution Analysis

### Statistical Characteristics
- **Mean Demand:** 17,744.30 units
- **Median Demand:** 2,000.00 units
- **Standard Deviation:** 47,608.76 units
- **Coefficient of Variation:** 2.68 (highly volatile)
- **Skewness:** 5.79 (highly right-skewed)
- **Kurtosis:** 46.83 (heavy-tailed distribution)

### Percentile Breakdown
| Percentile | Demand Value |
|------------|--------------|
| 10th       | 10 units     |
| 25th       | 150 units    |
| 50th       | 2,000 units  |
| 75th       | 10,000 units |
| 90th       | 50,000 units |
| 95th       | 100,000 units|
| 99th       | 230,200 units|

**Key Insights:**
- Extremely right-skewed distribution with median significantly lower than mean
- High variability indicates diverse product demand patterns
- Presence of high-demand outliers (5% of products account for disproportionate demand)

---

## ⏰ Temporal Patterns

### Yearly Trends
| Year | Records | Total Demand | Average Demand |
|------|---------|--------------|----------------|
| 2011 | 71      | 2,290,465    | 32,260.07     |
| 2012 | 7,917   | 139,240,115  | 17,587.48     |
| 2013 | 11      | 406,112      | 36,919.27     |

### Monthly Seasonality
**Peak Months:** May (21,201.40), August (21,891.76)
**Low Months:** September (13,778.37), Friday weekdays

| Month | Average Demand |
|-------|----------------|
| January | 15,700.02 |
| February | 16,161.33 |
| March | 15,804.14 |
| April | 18,553.00 |
| May | 21,201.40 |
| June | 17,223.97 |
| July | 18,610.09 |
| August | 21,891.76 |
| September | 13,778.37 |
| October | 17,973.99 |
| November | 17,819.88 |
| December | 18,595.32 |

### Weekly Patterns
**Weekend vs Weekday:**
- Weekend Average: 24,526.53 units
- Weekday Average: 17,553.39 units
- **Weekend/Weekday Ratio: 1.40** (40% higher weekend demand)

**Day-of-Week Breakdown:**
- **Saturday:** 49,115.00 (highest)
- **Tuesday:** 24,752.83
- **Sunday:** 23,350.04
- **Monday:** 18,978.16
- **Wednesday:** 14,303.07
- **Thursday:** 13,948.22
- **Friday:** 13,043.79 (lowest)

---

## 🏭 Business Dimensions Analysis

### Warehouse Performance
| Warehouse | Records | Total Demand | Average Demand | Market Share |
|-----------|---------|--------------|----------------|--------------|
| Whse_S    | 2,842   | 61,675,977   | 21,701.61     | 43.5%        |
| Whse_C    | 1,514   | 37,368,409   | 24,681.91     | 26.3%        |
| Whse_J    | 3,643   | 42,892,306   | 11,773.90     | 30.2%        |

**Key Insights:**
- Whse_S handles the highest volume despite fewer transactions
- Whse_C has the highest average demand per transaction
- Whse_J processes the most transactions but with lower average values

### Top-Performing Products
| Product Code | Total Demand | Transactions | Average Demand |
|--------------|--------------|--------------|----------------|
| Product_1359 | 18,265,000   | 454          | 40,231.28     |
| Product_1574 | 13,682,500   | 93           | 147,123.66    |
| Product_1245 | 11,319,000   | 50           | 226,380.00    |
| Product_1393 | 7,277,000    | 76           | 95,750.00     |
| Product_1341 | 6,315,000    | 103          | 61,310.68     |

### Top Product Categories
| Category | Total Demand | Transactions | Average Demand |
|----------|--------------|--------------|----------------|
| Category_019 | 121,799,652 | 4,271 | 28,517.83 |
| Category_006 | 10,971,546  | 322   | 34,073.12 |
| Category_005 | 4,356,620   | 780   | 5,585.41  |
| Category_030 | 2,108,500   | 199   | 10,595.48 |
| Category_033 | 2,040,000   | 29    | 70,344.83 |

**Category_019 dominates with 85.8% of total demand**

---

## ✅ Data Quality Assessment

### Completeness
- **Missing Values:** 1,040,576 records (99.2% of total dataset)
- **Complete Records:** 7,999 (0.8% of total dataset)
- **Zero Demand Records:** None in clean dataset
- **Duplicate Records:** Minimal

### Outlier Analysis
- **Potential Outliers (>95th percentile):** ~400 records
- **Maximum Demand Value:** 750,000 units
- **Outlier Impact:** Significant on mean calculations

---

## 🎯 Key Insights for Forecasting

### 1. Data Characteristics
- **High Volatility:** CV of 2.68 indicates extremely variable demand
- **Seasonal Patterns:** Clear monthly and weekly seasonality
- **Business Hierarchy:** Strong product and warehouse-specific patterns

### 2. Temporal Behavior
- **Weekend Effect:** 40% higher demand on weekends
- **Monthly Seasonality:** Peak demand in May and August
- **Weekly Cycles:** Saturday shows exceptional demand spikes

### 3. Business Patterns
- **Product Concentration:** Top 10 products account for ~50% of total demand
- **Category Dominance:** Category_019 represents majority of business
- **Warehouse Specialization:** Different demand profiles per warehouse

---

## 🚀 Recommendations for LSTM Forecasting

### 1. Data Preprocessing
- **Log Transformation:** Essential due to high skewness (5.79)
- **Outlier Handling:** Consider capping extreme values at 99th percentile
- **Missing Data Strategy:** Investigate the 99.2% missing data issue

### 2. Feature Engineering
- **Temporal Features:** Include day-of-week, month, weekend indicators
- **Lag Features:** Use 7-day and 30-day lags for seasonal patterns
- **Product Hierarchy:** Encode product categories and warehouse relationships
- **Rolling Statistics:** Moving averages to capture trends

### 3. Model Architecture Considerations
- **Multi-level Modeling:** Separate models for different product categories
- **Warehouse-Specific Models:** Account for distinct warehouse patterns
- **Ensemble Approaches:** Combine models for different demand ranges

### 4. Validation Strategy
- **Time-Based Splits:** Ensure temporal integrity in train/test splits
- **Weekend/Weekday Validation:** Separate evaluation for different day types
- **Category-Specific Metrics:** Evaluate performance by product category

### 5. External Factors to Consider
- **Holiday Calendar:** Incorporate retail holidays and special events
- **Promotional Activities:** Include marketing campaign indicators
- **Economic Indicators:** Seasonal retail patterns and economic cycles

---

## ⚠️ Identified Challenges

1. **Data Completeness:** 99.2% missing values require investigation
2. **High Variability:** CV of 2.68 makes forecasting challenging
3. **Outlier Impact:** Extreme values may distort model training
4. **Imbalanced Categories:** Category_019 dominance may bias results
5. **Limited Time Span:** Only 14 months of data for training

---

## 📊 Recommended Next Steps

1. **Data Investigation:** Analyze the source of 99.2% missing values
2. **Visualization Creation:** Generate plots for pattern confirmation
3. **Stationarity Testing:** Perform ADF tests on time series
4. **Correlation Analysis:** Examine feature relationships
5. **Model Baseline:** Establish simple forecasting benchmarks
6. **LSTM Enhancement:** Implement advanced preprocessing pipeline

This EDA provides a solid foundation for developing robust LSTM-based demand forecasting models with proper consideration of the data's unique characteristics and business context.