import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# Load prediction results
df = pd.read_csv("src/evaluation_results.csv")

# Ensure required columns exist
assert "Actual" in df.columns and "Predicted" in df.columns, "CSV must contain 'Actual' and 'Predicted' columns"

actual = df["Actual"]
predicted = df["Predicted"]

def test_mae():
    mae = mean_absolute_error(actual, predicted)
    print(f"MAE: {mae:.4f}")
    assert mae < 10, f"MAE too high: {mae}"

def test_rmse():
    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    print(f"RMSE: {rmse:.4f}")
    assert rmse < 10, f"RMSE too high: {rmse}"

def test_r2():
    r2 = r2_score(actual, predicted)
    print(f"R2: {r2:.4f}")
    assert r2 > 0.85, f"R2 too low: {r2}"

def test_mape():
    mape = mean_absolute_percentage_error(actual, predicted)
    print(f"MAPE: {mape:.4f}")
    assert mape < 0.2, f"MAPE too high: {mape}"
