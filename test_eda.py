import pandas as pd
import numpy as np

# Simple test function
def load_and_test():
    print("🔍 Testing basic data load...")
    
    # Update the file path to work in your environment
    filepath = "artifacts/Historical Product Demand.csv"
    
    try:
        df = pd.read_csv(filepath)
        print("✅ Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print("❌ File not found. Let's check what files are available...")
        import os
        print("Files in current directory:", os.listdir('.'))
        if os.path.exists('artifacts'):
            print("Files in artifacts:", os.listdir('artifacts'))
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# Simple main function
def main():
    print("🚀 Starting EDA test...")
    df = load_and_test()
    
    if df is not None:
        print(f"\n📊 Basic Info:")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
    else:
        print("Cannot proceed without data")

if __name__ == "__main__":
    main()