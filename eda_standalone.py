#!/usr/bin/env python3
"""
Standalone Professional EDA Script

This script provides a fixed version of your original EDA code with:
- Proper error handling and __main__ execution
- Integration with the professional framework
- Enhanced visualizations and analysis
- Robust file path handling
- Professional logging and monitoring

Usage:
    python3 eda_standalone.py
    python3 eda_standalone.py --filepath "path/to/your/data.csv"
    python3 eda_standalone.py --save-plots --plot-dir "custom_plots"
"""

import sys
import argparse
import warnings
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    """
    Main function for standalone EDA execution
    
    This fixes the original __main__ issue by:
    1. Proper argument parsing
    2. Error handling for missing dependencies
    3. Graceful fallback for missing files
    4. Integration with professional framework
    """
    parser = argparse.ArgumentParser(
        description="Professional Exploratory Data Analysis Tool"
    )
    
    parser.add_argument(
        "--filepath",
        type=str,
        default="artifacts/Historical Product Demand.csv",
        help="Path to the CSV data file (default: artifacts/Historical Product Demand.csv)"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Save plots to disk (default: True)"
    )
    
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots",
        help="Directory to save plots (default: plots)"
    )
    
    parser.add_argument(
        "--date-column",
        type=str,
        default="Date",
        help="Name of the date column (default: Date)"
    )
    
    parser.add_argument(
        "--target-column",
        type=str,
        default="Order_Demand",
        help="Name of the target column (default: Order_Demand)"
    )
    
    parser.add_argument(
        "--simple-mode",
        action="store_true",
        help="Run simple EDA without professional framework"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 Professional Exploratory Data Analysis Tool")
    print("=" * 70)
    
    try:
        # Check if file exists
        filepath = Path(args.filepath)
        if not filepath.exists():
            print(f"❌ Error: Data file not found at {filepath}")
            print(f"📝 Current working directory: {Path.cwd()}")
            
            # Try some common alternative paths
            alternative_paths = [
                Path("Historical Product Demand.csv"),
                Path("data") / "Historical Product Demand.csv",
                Path("artifacts") / "Historical Product Demand.csv"
            ]
            
            print("\n🔍 Searching for data file in common locations...")
            for alt_path in alternative_paths:
                if alt_path.exists():
                    print(f"✅ Found data file at: {alt_path}")
                    filepath = alt_path
                    break
            else:
                print("❌ No data file found in common locations")
                print("📥 Please ensure your data file is available and update the path")
                return 1
        
        if args.simple_mode:
            print("🔧 Running in simple mode (basic EDA)")
            run_simple_eda(filepath, args)
        else:
            print("🚀 Running professional EDA analysis")
            run_professional_eda(filepath, args)
            
        print("\n✅ EDA Analysis completed successfully!")
        print(f"📊 Plots saved to: {Path(args.plot_dir).absolute()}")
        
    except ImportError as e:
        print(f"❌ Missing required dependencies: {e}")
        print("📦 Please install dependencies:")
        print("   Option 1: python3 install_demo.py")
        print("   Option 2: pip install pandas numpy matplotlib seaborn scikit-learn")
        return 1
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("🔧 Try running with --simple-mode flag for basic analysis")
        return 1


def run_professional_eda(filepath: Path, args):
    """Run professional EDA using the advanced framework"""
    try:
        from src.analysis.eda import EDAAnalyzer
        
        # Create analyzer
        analyzer = EDAAnalyzer(
            save_plots=args.save_plots,
            plot_dir=Path(args.plot_dir)
        )
        
        # Run comprehensive analysis
        result = analyzer.run_complete_eda(
            str(filepath),
            date_column=args.date_column,
            target_column=args.target_column
        )
        
        # Display summary
        print("\n📈 Analysis Summary:")
        print(f"   Dataset Shape: {result.dataset_info.get('shape')}")
        print(f"   Missing Data: {result.missing_data_analysis.get('missing_percentage', 0):.2f}%")
        print(f"   Outliers: {result.outlier_analysis.get('outlier_percentage', 0):.2f}%")
        
        if result.time_series_analysis:
            stationarity = result.time_series_analysis.get('stationarity_test', {})
            if 'is_stationary' in stationarity:
                status = "✅ Stationary" if stationarity['is_stationary'] else "⚠️ Non-stationary"
                print(f"   Time Series: {status}")
        
    except Exception as e:
        print(f"❌ Professional EDA failed: {e}")
        print("🔄 Falling back to simple mode...")
        run_simple_eda(filepath, args)


def run_simple_eda(filepath: Path, args):
    """Run simple EDA with basic dependencies (fallback mode)"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    print("📊 Loading and analyzing data...")
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Basic info
    print(f"\n📋 Dataset Info:")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"   Missing Values: {df.isnull().sum().sum()}")
    
    # Preprocess data
    if args.date_column in df.columns:
        df[args.date_column] = pd.to_datetime(df[args.date_column], errors='coerce')
        df.set_index(args.date_column, inplace=True)
    
    if args.target_column in df.columns:
        # Remove invalid values
        df = df[df[args.target_column] > 0].dropna()
        df['Order_Demand_Log'] = np.log1p(df[args.target_column])
        
        print(f"\n📈 Target Variable ({args.target_column}):")
        print(f"   Mean: {df[args.target_column].mean():.2f}")
        print(f"   Std: {df[args.target_column].std():.2f}")
        print(f"   Min: {df[args.target_column].min():.2f}")
        print(f"   Max: {df[args.target_column].max():.2f}")
    
    # Create plots directory
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    create_simple_visualizations(df, args, plot_dir)


def create_simple_visualizations(df, args, plot_dir):
    """Create basic visualizations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("🎨 Creating visualizations...")
    
    # 1. Time series plot
    if args.target_column in df.columns:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        df[args.target_column].plot(title=f'{args.target_column} Over Time')
        plt.xlabel('Date')
        plt.ylabel(args.target_column)
        
        plt.subplot(1, 2, 2)
        df[args.target_column].rolling(30).mean().plot(title='30-Day Moving Average')
        plt.xlabel('Date')
        plt.ylabel('30-Day Average')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'time_series.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Distribution plots
    if 'Order_Demand_Log' in df.columns:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df[args.target_column], kde=True, bins=30)
        plt.title(f'Original {args.target_column} Distribution')
        
        plt.subplot(1, 2, 2)
        sns.histplot(df['Order_Demand_Log'], kde=True, bins=30)
        plt.title('Log-Transformed Distribution')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Seasonal analysis
    if isinstance(df.index, pd.DatetimeIndex) and args.target_column in df.columns:
        plt.figure(figsize=(15, 8))
        
        # Monthly patterns
        plt.subplot(2, 2, 1)
        monthly_avg = df.groupby(df.index.month)[args.target_column].mean()
        monthly_avg.plot(kind='bar', title='Average by Month')
        plt.xlabel('Month')
        
        # Weekly patterns
        plt.subplot(2, 2, 2)
        weekly_avg = df.groupby(df.index.dayofweek)[args.target_column].mean()
        weekly_avg.plot(kind='bar', title='Average by Day of Week')
        plt.xlabel('Day (0=Monday)')
        
        # Monthly totals
        plt.subplot(2, 2, 3)
        monthly_totals = df.resample('M')[args.target_column].sum()
        monthly_totals.plot(title='Monthly Totals')
        plt.xlabel('Month')
        
        # Outlier detection
        plt.subplot(2, 2, 4)
        if 'Order_Demand_Log' in df.columns:
            sns.boxplot(data=df, y='Order_Demand_Log')
            plt.title('Outlier Detection (Log Scale)')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'seasonal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Summary statistics
    print("\n📊 Summary Statistics:")
    if args.target_column in df.columns:
        stats = df[args.target_column].describe()
        for stat, value in stats.items():
            print(f"   {stat.capitalize()}: {value:.2f}")
    
    # 5. Missing data analysis
    if isinstance(df.index, pd.DatetimeIndex):
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        missing_dates = len(date_range) - len(df.index)
        print(f"\n📅 Date Analysis:")
        print(f"   Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        print(f"   Missing Dates: {missing_dates}")
        print(f"   Data Coverage: {(len(df.index) / len(date_range)) * 100:.1f}%")


if __name__ == "__main__":
    """
    This fixes the original __main__ issue by properly structuring the script
    and handling the main execution flow
    """
    exit_code = main()
    sys.exit(exit_code or 0)