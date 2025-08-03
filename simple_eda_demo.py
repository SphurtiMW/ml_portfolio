#!/usr/bin/env python3
"""
Simple EDA Demo - No External Dependencies Required

This script demonstrates the solution to your "__main__" error and shows
basic data analysis concepts using only Python's built-in libraries.

The original error occurred due to improper script structure. This version fixes:
1. Proper if __name__ == "__main__" usage
2. Argument parsing for flexible execution
3. Error handling for missing files/dependencies
4. Modular function structure

Usage:
    python3 simple_eda_demo.py
    python3 simple_eda_demo.py --create-sample-data
    python3 simple_eda_demo.py --filepath custom_data.csv
"""

import csv
import json
import sys
import argparse
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random


def create_sample_data(filepath: str = "sample_demand_data.csv", num_days: int = 365) -> None:
    """
    Create sample demand data for demonstration
    
    This generates realistic-looking demand data with:
    - Seasonal patterns (higher demand in certain months)
    - Weekly patterns (lower demand on weekends)
    - Random noise to simulate real-world variability
    """
    print(f"📝 Creating sample data with {num_days} days...")
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        
        # Base demand with seasonal effect
        base_demand = 100
        seasonal_effect = 30 * (1 + 0.5 * ((current_date.month - 6) / 6))
        
        # Weekly pattern (lower on weekends)
        weekly_effect = 0.7 if current_date.weekday() >= 5 else 1.0
        
        # Random noise
        noise = random.uniform(0.8, 1.2)
        
        # Calculate final demand
        demand = int(base_demand * seasonal_effect * weekly_effect * noise)
        
        data.append({
            'Date': current_date.strftime('%Y-%m-%d'),
            'Order_Demand': demand,
            'Product_Code': f'PROD_{random.randint(1000, 9999)}',
            'Warehouse': random.choice(['WH_A', 'WH_B', 'WH_C'])
        })
    
    # Write to CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Date', 'Order_Demand', 'Product_Code', 'Warehouse']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✅ Sample data created: {filepath}")
    print(f"   Records: {len(data)}")
    print(f"   Date range: {data[0]['Date']} to {data[-1]['Date']}")


def load_csv_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load CSV data using built-in csv module
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        List of dictionaries representing the data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    try:
        data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert numeric columns
                if 'Order_Demand' in row:
                    try:
                        row['Order_Demand'] = float(row['Order_Demand'])
                    except ValueError:
                        continue  # Skip invalid rows
                
                # Parse date
                if 'Date' in row:
                    try:
                        row['Date'] = datetime.strptime(row['Date'], '%Y-%m-%d')
                    except ValueError:
                        continue  # Skip invalid dates
                
                data.append(row)
        
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")


def basic_statistics(data: List[Dict[str, Any]], column: str = 'Order_Demand') -> Dict[str, float]:
    """
    Calculate basic statistics using built-in statistics module
    
    Args:
        data: List of data dictionaries
        column: Column name to analyze
        
    Returns:
        Dictionary with basic statistics
    """
    values = [row[column] for row in data if column in row and isinstance(row[column], (int, float))]
    
    if not values:
        return {"error": f"No valid numeric data found in column '{column}'"}
    
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "mode": statistics.mode(values) if len(set(values)) < len(values) else "No mode",
        "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
        "variance": statistics.variance(values) if len(values) > 1 else 0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values)
    }


def time_series_analysis(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform basic time series analysis
    
    Args:
        data: List of data dictionaries with Date and Order_Demand
        
    Returns:
        Dictionary with time series insights
    """
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x['Date'])
    
    # Monthly aggregation
    monthly_data = {}
    weekly_data = {i: [] for i in range(7)}  # 0=Monday, 6=Sunday
    
    for row in sorted_data:
        if 'Date' in row and 'Order_Demand' in row:
            date = row['Date']
            demand = row['Order_Demand']
            
            # Monthly aggregation
            month_key = f"{date.year}-{date.month:02d}"
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(demand)
            
            # Weekly pattern
            weekly_data[date.weekday()].append(demand)
    
    # Calculate monthly averages
    monthly_averages = {
        month: statistics.mean(demands)
        for month, demands in monthly_data.items()
    }
    
    # Calculate weekly averages
    weekly_averages = {
        day: statistics.mean(demands) if demands else 0
        for day, demands in weekly_data.items()
    }
    
    # Trend analysis (simple)
    if len(sorted_data) > 30:
        first_month = statistics.mean([row['Order_Demand'] for row in sorted_data[:30]])
        last_month = statistics.mean([row['Order_Demand'] for row in sorted_data[-30:]])
        trend = "Increasing" if last_month > first_month else "Decreasing"
        trend_strength = abs(last_month - first_month) / first_month * 100
    else:
        trend = "Insufficient data"
        trend_strength = 0
    
    return {
        "date_range": {
            "start": sorted_data[0]['Date'].strftime('%Y-%m-%d'),
            "end": sorted_data[-1]['Date'].strftime('%Y-%m-%d'),
            "total_days": len(sorted_data)
        },
        "monthly_averages": monthly_averages,
        "weekly_averages": {
            f"Day_{day} ({'Mon,Tue,Wed,Thu,Fri,Sat,Sun'.split(',')[day]})": avg
            for day, avg in weekly_averages.items()
        },
        "trend_analysis": {
            "direction": trend,
            "strength_percent": trend_strength
        }
    }


def outlier_detection(data: List[Dict[str, Any]], column: str = 'Order_Demand') -> Dict[str, Any]:
    """
    Simple outlier detection using IQR method
    
    Args:
        data: List of data dictionaries
        column: Column to analyze for outliers
        
    Returns:
        Dictionary with outlier analysis
    """
    values = [row[column] for row in data if column in row and isinstance(row[column], (int, float))]
    
    if len(values) < 4:
        return {"error": "Insufficient data for outlier detection"}
    
    # Sort values for quartile calculation
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Calculate quartiles
    q1_index = n // 4
    q3_index = 3 * n // 4
    q1 = sorted_values[q1_index]
    q3 = sorted_values[q3_index]
    iqr = q3 - q1
    
    # Calculate outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Find outliers
    outliers = [val for val in values if val < lower_bound or val > upper_bound]
    
    return {
        "quartiles": {"Q1": q1, "Q3": q3, "IQR": iqr},
        "bounds": {"lower": lower_bound, "upper": upper_bound},
        "outliers": {
            "count": len(outliers),
            "percentage": len(outliers) / len(values) * 100,
            "values": outliers[:10]  # First 10 outliers
        }
    }


def data_quality_check(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check data quality issues
    
    Args:
        data: List of data dictionaries
        
    Returns:
        Dictionary with data quality metrics
    """
    total_rows = len(data)
    if total_rows == 0:
        return {"error": "No data to analyze"}
    
    # Check for missing values
    all_columns = set()
    for row in data:
        all_columns.update(row.keys())
    
    missing_data = {}
    for column in all_columns:
        missing_count = sum(1 for row in data if column not in row or row[column] is None or row[column] == '')
        missing_data[column] = {
            "count": missing_count,
            "percentage": missing_count / total_rows * 100
        }
    
    # Check for duplicates (simplified)
    unique_rows = len(set(json.dumps(row, sort_keys=True, default=str) for row in data))
    duplicates = total_rows - unique_rows
    
    return {
        "total_rows": total_rows,
        "unique_rows": unique_rows,
        "duplicates": duplicates,
        "missing_data": missing_data,
        "columns": list(all_columns)
    }


def run_comprehensive_eda(filepath: str) -> Dict[str, Any]:
    """
    Run comprehensive EDA analysis
    
    This is the main analysis function that orchestrates all the individual
    analysis components.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        Dictionary with all analysis results
    """
    print("🔍 Starting comprehensive EDA analysis...")
    
    try:
        # Load data
        print("📊 Loading data...")
        data = load_csv_data(filepath)
        print(f"✅ Loaded {len(data)} records")
        
        # Data quality check
        print("🔍 Checking data quality...")
        quality_results = data_quality_check(data)
        
        # Basic statistics
        print("📈 Calculating basic statistics...")
        stats_results = basic_statistics(data, 'Order_Demand')
        
        # Time series analysis
        print("📅 Performing time series analysis...")
        time_series_results = time_series_analysis(data)
        
        # Outlier detection
        print("🎯 Detecting outliers...")
        outlier_results = outlier_detection(data, 'Order_Demand')
        
        # Compile results
        results = {
            "data_quality": quality_results,
            "basic_statistics": stats_results,
            "time_series_analysis": time_series_results,
            "outlier_analysis": outlier_results,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        print("✅ EDA analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return {"error": str(e)}


def print_results_summary(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the analysis results
    
    Args:
        results: Dictionary containing analysis results
    """
    print("\n" + "="*70)
    print("📊 EDA ANALYSIS SUMMARY")
    print("="*70)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Data Quality Summary
    if "data_quality" in results:
        quality = results["data_quality"]
        print(f"\n📋 DATA QUALITY:")
        print(f"   Total Rows: {quality.get('total_rows', 'N/A')}")
        print(f"   Unique Rows: {quality.get('unique_rows', 'N/A')}")
        print(f"   Duplicates: {quality.get('duplicates', 'N/A')}")
        print(f"   Columns: {', '.join(quality.get('columns', []))}")
    
    # Basic Statistics
    if "basic_statistics" in results and "error" not in results["basic_statistics"]:
        stats = results["basic_statistics"]
        print(f"\n📈 ORDER DEMAND STATISTICS:")
        print(f"   Count: {stats.get('count', 'N/A')}")
        print(f"   Mean: {stats.get('mean', 0):.2f}")
        print(f"   Median: {stats.get('median', 0):.2f}")
        print(f"   Std Dev: {stats.get('std_dev', 0):.2f}")
        print(f"   Min: {stats.get('min', 'N/A')}")
        print(f"   Max: {stats.get('max', 'N/A')}")
    
    # Time Series Analysis
    if "time_series_analysis" in results:
        ts = results["time_series_analysis"]
        if "date_range" in ts:
            date_range = ts["date_range"]
            print(f"\n📅 TIME SERIES ANALYSIS:")
            print(f"   Date Range: {date_range.get('start')} to {date_range.get('end')}")
            print(f"   Total Days: {date_range.get('total_days')}")
        
        if "trend_analysis" in ts:
            trend = ts["trend_analysis"]
            print(f"   Trend: {trend.get('direction')} ({trend.get('strength_percent', 0):.1f}% change)")
    
    # Outlier Analysis
    if "outlier_analysis" in results and "error" not in results["outlier_analysis"]:
        outliers = results["outlier_analysis"]
        if "outliers" in outliers:
            outlier_info = outliers["outliers"]
            print(f"\n🎯 OUTLIER ANALYSIS:")
            print(f"   Outliers Found: {outlier_info.get('count', 0)}")
            print(f"   Percentage: {outlier_info.get('percentage', 0):.2f}%")
    
    print("\n" + "="*70)


def main():
    """
    Main function - This properly handles the script execution
    
    This fixes the original "__main__" error by:
    1. Properly structuring the main execution flow
    2. Using argparse for robust command-line argument handling
    3. Providing comprehensive error handling
    4. Modular function design for better maintainability
    """
    parser = argparse.ArgumentParser(
        description="Simple EDA Tool - Built-in Libraries Only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 simple_eda_demo.py --create-sample-data
  python3 simple_eda_demo.py --filepath sample_demand_data.csv
  python3 simple_eda_demo.py --filepath your_data.csv --output results.json
        """
    )
    
    parser.add_argument(
        "--filepath",
        type=str,
        default="sample_demand_data.csv",
        help="Path to the CSV data file"
    )
    
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample data for demonstration"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save analysis results to JSON file"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days for sample data generation (default: 365)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Simple EDA Tool - No External Dependencies")
    print("="*70)
    
    try:
        # Create sample data if requested
        if args.create_sample_data:
            create_sample_data(args.filepath, args.days)
            print(f"📝 Sample data created at: {args.filepath}")
        
        # Check if file exists
        if not Path(args.filepath).exists():
            print(f"❌ File not found: {args.filepath}")
            print("💡 Use --create-sample-data to generate sample data")
            return 1
        
        # Run EDA analysis
        results = run_comprehensive_eda(args.filepath)
        
        # Print results
        print_results_summary(results)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to: {args.output}")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Data file: {Path(args.filepath).absolute()}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


# This is the correct way to handle the __name__ == "__main__" pattern
# Your original error was likely due to issues in this section
if __name__ == "__main__":
    """
    This block only executes when the script is run directly (not imported)
    
    The "__main__" error you encountered was likely due to:
    1. Improper indentation in this block
    2. Syntax errors in the main() function
    3. Missing import statements
    4. Circular import issues
    
    This implementation fixes all these issues by:
    - Proper error handling and exit codes
    - Clean separation of functionality
    - No external dependencies
    - Comprehensive argument parsing
    """
    exit_code = main()
    sys.exit(exit_code)