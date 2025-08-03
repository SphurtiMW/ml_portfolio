#!/usr/bin/env python3
"""
Simple Analysis to Show Missing EDA Components
"""

import csv
from datetime import datetime, timedelta
from collections import defaultdict
import math

def load_data(filename):
    """Load data from CSV"""
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Order_Demand'] and row['Order_Demand'].strip():
                data.append({
                    'date': datetime.strptime(row['Date'], '%Y-%m-%d'),
                    'demand': float(row['Order_Demand'])
                })
    return data

def analyze_missing_patterns(data):
    """Analyze missing data patterns"""
    print("=== MISSING DATA ANALYSIS ===")
    
    # Find missing dates
    all_dates = set()
    data_dates = set()
    
    for entry in data:
        data_dates.add(entry['date'].date())
    
    start_date = min(entry['date'] for entry in data).date()
    end_date = max(entry['date'] for entry in data).date()
    
    current = start_date
    while current <= end_date:
        all_dates.add(current)
        current = (datetime.combine(current, datetime.min.time()) + timedelta(days=1)).date()
    
    missing_dates = all_dates - data_dates
    print(f"Missing dates: {len(missing_dates)}")
    print(f"Data completeness: {len(data_dates)/len(all_dates)*100:.1f}%")
    
    return data

def analyze_weekly_patterns(data):
    """Analyze weekly patterns"""
    print("\n=== WEEKLY PATTERNS ===")
    
    weekly_demands = defaultdict(list)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for entry in data:
        day_of_week = entry['date'].weekday()
        weekly_demands[day_of_week].append(entry['demand'])
    
    print("Average demand by day of week:")
    for i, day in enumerate(day_names):
        if i in weekly_demands:
            avg_demand = sum(weekly_demands[i]) / len(weekly_demands[i])
            print(f"{day}: {avg_demand:.1f}")
    
    return data

def analyze_monthly_patterns(data):
    """Analyze monthly patterns"""
    print("\n=== MONTHLY PATTERNS ===")
    
    monthly_demands = defaultdict(list)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for entry in data:
        month = entry['date'].month
        monthly_demands[month].append(entry['demand'])
    
    print("Average demand by month:")
    for i, month in enumerate(month_names):
        if i+1 in monthly_demands:
            avg_demand = sum(monthly_demands[i+1]) / len(monthly_demands[i+1])
            print(f"{month}: {avg_demand:.1f}")
    
    return data

def analyze_trends(data):
    """Analyze trends"""
    print("\n=== TREND ANALYSIS ===")
    
    # Sort by date
    sorted_data = sorted(data, key=lambda x: x['date'])
    
    # Simple linear trend
    n = len(sorted_data)
    x_values = list(range(n))
    y_values = [entry['demand'] for entry in sorted_data]
    
    # Calculate trend
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    
    print(f"Trend slope: {slope:.2f}")
    if slope > 0:
        print("Trend: Increasing demand over time")
    elif slope < 0:
        print("Trend: Decreasing demand over time")
    else:
        print("Trend: No significant trend")
    
    return data

def analyze_outliers(data):
    """Analyze outliers"""
    print("\n=== OUTLIER ANALYSIS ===")
    
    demands = [entry['demand'] for entry in data]
    demands.sort()
    
    n = len(demands)
    q1_idx = int(0.25 * n)
    q3_idx = int(0.75 * n)
    
    q1 = demands[q1_idx]
    q3 = demands[q3_idx]
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = [d for d in demands if d < lower_bound or d > upper_bound]
    
    print(f"Q1: {q1:.1f}")
    print(f"Q3: {q3:.1f}")
    print(f"IQR: {iqr:.1f}")
    print(f"Outliers: {len(outliers)} ({len(outliers)/len(demands)*100:.1f}%)")
    
    if outliers:
        print(f"Outlier values: {outliers}")
    
    return data

def generate_recommendations(data):
    """Generate recommendations"""
    print("\n=== RECOMMENDATIONS ===")
    
    demands = [entry['demand'] for entry in data]
    
    # Check data characteristics
    print(f"Data points: {len(data)}")
    print(f"Date range: {min(entry['date'] for entry in data)} to {max(entry['date'] for entry in data)}")
    
    # Check for seasonality
    weekly_vars = defaultdict(list)
    for entry in data:
        day_of_week = entry['date'].weekday()
        weekly_vars[day_of_week].append(entry['demand'])
    
    weekly_variance = sum([(sum(vals)/len(vals))**2 for vals in weekly_vars.values()]) / len(weekly_vars)
    
    print(f"\nWeekly pattern variance: {weekly_variance:.2f}")
    
    print("\nRECOMMENDATIONS:")
    print("✅ Proceed with model training")
    print("✅ Include day-of-week features")
    print("✅ Consider seasonal decomposition")
    print("✅ Test ARIMA and Prophet models")
    print("✅ Use time series cross-validation")

def main():
    """Main function"""
    print("SIMPLE EDA ANALYSIS")
    print("=" * 40)
    
    try:
        data = load_data("Historical Product Demand.csv")
        print(f"Loaded {len(data)} data points")
        
        data = analyze_missing_patterns(data)
        data = analyze_weekly_patterns(data)
        data = analyze_monthly_patterns(data)
        data = analyze_trends(data)
        data = analyze_outliers(data)
        generate_recommendations(data)
        
        print("\n" + "=" * 40)
        print("ANALYSIS COMPLETED!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()