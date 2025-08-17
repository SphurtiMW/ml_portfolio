import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# ---- CONFIGURATION ----
DATA_PATH = "artifacts/Historical Product Demand.csv"
SAVE_PLOT_PATH = "artifacts/sarima_forecast_plot.png"
SAVE_CSV_PATH = "artifacts/sarima_forecast.csv"
FORECAST_DAYS = 14

# ---- LOAD & PREPROCESS ----
df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date", low_memory=False)
df = df[['Order_Demand']].copy()
df = df[~df.index.duplicated(keep='first')].sort_index()

# Log-transform to stabilize variance
df['Order_Demand_Log'] = np.log1p(df['Order_Demand'])

# ---- TRAINING SARIMA ----
# You can tune (p,d,q)x(P,D,Q,s) further
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

model = SARIMAX(df['Order_Demand_Log'], order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

# ---- FORECASTING ----
forecast_log = results.get_forecast(steps=FORECAST_DAYS).predicted_mean
forecast = np.expm1(forecast_log)  # Inverse of log1p

forecast_df = pd.DataFrame({
    "Date": pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='D'),
    "SARIMA_Prediction": forecast.values
})
forecast_df.set_index("Date", inplace=True)

# ---- SAVE FORECAST ----
os.makedirs("artifacts", exist_ok=True)
forecast_df.to_csv(SAVE_CSV_PATH)

# ---- PLOT ----
plt.figure(figsize=(10, 5))
plt.plot(forecast_df.index, forecast_df['SARIMA_Prediction'], marker='o', linestyle='-')
plt.title("SARIMA 14-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Demand")
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_PLOT_PATH)
plt.show()

print("SARIMA training and forecasting complete.")
print(f"Forecast saved to: {SAVE_CSV_PATH}")
print(f"Plot saved to: {SAVE_PLOT_PATH}")
