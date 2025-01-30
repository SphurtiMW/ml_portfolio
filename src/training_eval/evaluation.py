import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ============================
# Step 1: Load the trained model
# ============================
MODEL_PATH = r"C:\Users\sphur\OneDrive\Desktop\ml_portfolio\src\components\lstm_demand_forecasting.h5"
DATA_PATH = r"C:\Users\sphur\Downloads\Historical Product Demand.csv"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

print("Loading trained model...")
model = load_model(MODEL_PATH, safe_mode=False)  # Use safe_mode=False if needed
print("Model loaded successfully!")

# ============================
# Step 2: Load and preprocess the dataset
# ============================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")

# Select relevant column
df = df[['Order_Demand']]

# Apply log transformation (same as training)
df['Order_Demand_Log'] = np.log1p(df['Order_Demand'])

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

scaler.fit(train_data[['Order_Demand_Log']])
df_scaled = scaler.transform(df[['Order_Demand_Log']])

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])  
    return np.array(X), np.array(y)

# Ensure the sequence length matches the trained model
SEQ_LENGTH = 30  # Fix the mismatch

# Generate test sequences
X, y = create_sequences(df_scaled, SEQ_LENGTH)

# Split into test set
X_test, y_test = X[train_size:], y[train_size:]

# Reshape for LSTM input
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Verify input shape before making predictions
print(f"X_test shape: {X_test.shape}, Model expected: {model.input_shape}")

# ============================
# Step 3: Make Predictions
# ============================
print("Making predictions...")
y_pred = model.predict(X_test)

# Convert NaN values to zero before inverse transformation
y_pred = np.nan_to_num(y_pred)
y_test = np.nan_to_num(y_test)

# Inverse transform to get actual values
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

print("Predictions complete!")

# ============================
# Step 4: Check for NaN values in predictions
# ============================
print("Checking for NaN values in evaluation data...")
if np.isnan(y_test_actual).any():
    print("Warning: NaN values found in y_test_actual.")
if np.isnan(y_pred_actual).any():
    print("Warning: NaN values found in y_pred_actual.")

# Remove NaN values before evaluation
y_test_actual = y_test_actual[~np.isnan(y_test_actual)]
y_pred_actual = y_pred_actual[~np.isnan(y_pred_actual)]

print("NaN values removed. Proceeding with evaluation...")

# ============================
# Step 5: Evaluate Model Performance
# ============================
print("Evaluating model performance...")

mse = mean_squared_error(y_test_actual, y_pred_actual)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ============================
# Step 6: Visualize Predictions vs. Actual
# ============================
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label="Actual Demand", color="blue")
plt.plot(y_pred_actual, label="Predicted Demand", color="red", linestyle="dashed")
plt.title("LSTM Demand Forecasting - Model Evaluation")
plt.xlabel("Time")
plt.ylabel("Order Demand")
plt.legend()
plt.show()

print("Evaluation complete. Check the plotted graph for results.")
