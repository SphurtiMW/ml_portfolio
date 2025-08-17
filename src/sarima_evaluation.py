import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("artifacts/Historical Product Demand.csv", parse_dates=["Date"])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

# Aggregate by date (some dates may have multiple entries)
df_daily = df.groupby(df.index).sum()
df_daily = df_daily.asfreq('D').fillna(method='ffill')  # Ensure daily frequency

# Train-test split: last 14 days for test
train = df_daily.iloc[:-14]
test = df_daily.iloc[-14:]

# Train SARIMA model (adjust params if needed)
sarima_model = SARIMAX(
    train["Order_Demand"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# Forecast
forecast = sarima_model.get_forecast(steps=14)
forecast_mean = forecast.predicted_mean
forecast_index = test.index

# Evaluation metrics
actual = test["Order_Demand"]
mae = mean_absolute_error(actual, forecast_mean)
rmse = np.sqrt(mean_squared_error(actual, forecast_mean))
mape = np.mean(np.abs((actual - forecast_mean) / actual)) * 100

print("\n--- SARIMA Forecast Evaluation ---")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(train.index[-30:], train["Order_Demand"].iloc[-30:], label="Recent Demand (Train)")
plt.plot(actual.index, actual.values, label="Actual Demand (Test)", marker='o')
plt.plot(forecast_index, forecast_mean.values, label="SARIMA Forecast", linestyle='--', marker='x')
plt.title("SARIMA: Actual vs Forecast (Next 14 Days)")
plt.xlabel("Date")
plt.ylabel("Order Demand")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("artifacts/sarima_eval_plot.png")
plt.show()
