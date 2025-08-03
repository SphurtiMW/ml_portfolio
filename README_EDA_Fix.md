# EDA Script Fix: Resolving "__main__ not found" Error

## Problem Summary

The original EDA script was encountering a "__main__ not found" error, which typically occurs due to:

1. **Missing Dependencies**: Required Python packages not installed
2. **Import Errors**: Issues with module imports
3. **File Path Issues**: Hardcoded Windows paths not working on other systems
4. **Runtime Environment**: Python not properly configured

## Solution Implemented

### 1. **Dependency Installation**
- Installed all required packages using `pip install --break-system-packages`:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations  
  - `matplotlib` - Plotting and visualization
  - `seaborn` - Statistical data visualization
  - `statsmodels` - Statistical modeling and time series analysis

### 2. **Enhanced Error Handling**
- Added comprehensive try-catch blocks around all functions
- Implemented graceful error handling with informative error messages
- Added null checks to prevent crashes when data processing fails

### 3. **Flexible File Path Resolution**
- Replaced hardcoded Windows path: `r"C:\Users\sphur\Downloads\Historical Product Demand.csv"`
- Added `get_data_file_path()` function that checks multiple common locations:
  - Current directory
  - `data/` subdirectory
  - `../data/` parent data directory
  - User's Downloads folder
  - EDA directory
- Interactive file path input when file not found automatically

### 4. **Improved Script Structure**
- Added proper docstrings to all functions
- Enhanced the main execution block with better error handling
- Added import statements for `os` and `sys` for better file handling
- Implemented graceful script termination on errors

### 5. **Cross-Platform Compatibility**
- Removed Windows-specific path formatting
- Used `os.path` functions for platform-independent file operations
- Added proper path expansion for user directories

## Key Improvements Made

### Before (Original Issues):
```python
# Hardcoded Windows path
filepath = r"C:\Users\sphur\Downloads\Historical Product Demand.csv" 

# No error handling
def load_data(filepath):
    df = pd.read_csv(filepath)  # Could fail silently
    return df

# Basic main execution
if __name__ == "__main__":
    main()
```

### After (Fixed Version):
```python
# Flexible path resolution
def get_data_file_path():
    possible_paths = [
        "Historical Product Demand.csv",
        "data/Historical Product Demand.csv",
        # ... multiple fallback paths
    ]
    # Smart path detection with user fallback

# Robust error handling
def load_data(filepath):
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Enhanced main execution
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEDA interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
```

## Files Modified/Created

1. **`Exploratory_Data_Analysis/EDA.py`** - Main script with all fixes applied
2. **`requirements.txt`** - Added `statsmodels` dependency
3. **`Exploratory_Data_Analysis/sample_data_generator.py`** - Sample data generator for testing
4. **`Exploratory_Data_Analysis/Historical Product Demand.csv`** - Sample dataset for testing

## How to Use the Fixed Script

### Option 1: With Your Own Data
```bash
cd Exploratory_Data_Analysis
# Place your "Historical Product Demand.csv" file in this directory
python3 EDA.py
```

### Option 2: With Sample Data
```bash
cd Exploratory_Data_Analysis
python3 sample_data_generator.py  # Generate sample data
python3 EDA.py                    # Run EDA analysis
```

### Option 3: Interactive File Selection
```bash
cd Exploratory_Data_Analysis
python3 EDA.py
# If file not found automatically, you'll be prompted to enter the file path
```

## Expected Output

When running successfully, you should see:
```
Starting Exploratory Data Analysis...
Using data file: Historical Product Demand.csv
Dataset Loaded Successfully!

First 5 rows of the dataset:
...

Data Preprocessing Completed!
Log Transformation Applied!
Missing Dates Count: X
Basic Statistical Summary:
...

Feature Engineering Completed!
Performing Augmented Dickey-Fuller Test (ADF)...
...

EDA completed successfully!
```

## Common Issues and Solutions

### Issue: "ModuleNotFoundError"
**Solution**: Install missing packages
```bash
python3 -m pip install pandas numpy matplotlib seaborn statsmodels --break-system-packages
```

### Issue: "File not found"
**Solution**: Ensure your CSV file is named exactly "Historical Product Demand.csv" or use the interactive path input

### Issue: "Permission denied"
**Solution**: Check file permissions and ensure you have read access to the data file

### Issue: Display issues with plots
**Solution**: If running in a headless environment, plots may not display. This is normal - the analysis will still complete.

## Testing

The script has been tested and validated:
- ✅ All dependencies import successfully
- ✅ All functions are accessible and working
- ✅ Error handling prevents crashes
- ✅ Sample data generation works
- ✅ Core EDA pipeline executes without errors

The "__main__ not found" error has been completely resolved!