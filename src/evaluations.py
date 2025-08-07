import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load prediction results (ensure this file exists)
df = pd.read_csv("src/predictions.csv")

# Rename columns if needed
# You can uncomment and adjust below if the CSV has different headers
# df.columns = ['Actual', 'Predicted']

# Check for required columns
if 'Actual' not in df.columns or 'Predicted' not in df.columns:
    raise ValueError("CSV must have 'Actual' and 'Predicted' columns.")

# Metrics
mae = mean_absolute_error(df['Actual'], df['Predicted'])
rmse = mean_squared_error(df['Actual'], df['Predicted'], squared=False)
r2 = r2_score(df['Actual'], df['Predicted'])
mape = mean_absolute_percentage_error(df['Actual'], df['Predicted']) * 100

metrics = {
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "R2 Score": round(r2, 4),
    "MAPE": round(mape, 2)
}

print("\nEvaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Save metrics to JSON
with open("evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save results as CSV for test suite
df.to_csv("evaluation_results.csv", index=False)

# Optional: Save actual vs predicted plot
plt.figure(figsize=(10, 5))
plt.plot(df['Actual'].values, label='Actual', marker='o')
plt.plot(df['Predicted'].values, label='Predicted', marker='x')
plt.title("Actual vs Predicted")
plt.xlabel("Samples")
plt.ylabel("Order Demand")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation_plot.png")
plt.close()
