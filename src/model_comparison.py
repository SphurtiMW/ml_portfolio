import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Ensure the output directory exists
os.makedirs("artifacts", exist_ok=True)

# Define the metrics and model scores
metrics = ['MAE', 'RMSE', 'MAPE']
lstm_scores = [6.50, 8.50, 1.41]
sarima_scores = [45.57, 99.84, 19.17]

# Bar Plot
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bar1 = ax.bar(x - width/2, lstm_scores, width, label='LSTM', color='tab:blue')
bar2 = ax.bar(x + width/2, sarima_scores, width, label='SARIMA', color='tab:orange')

ax.set_ylabel('Error Value')
ax.set_title('LSTM vs SARIMA - Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

bar_chart_path = "artifacts/lstm_vs_sarima_bar.png"
plt.savefig(bar_chart_path)
plt.close()

print(f"Bar chart saved to: {bar_chart_path}")

# Radar Chart
# Normalize for radar chart (0 = best, 1 = worst)
max_vals = [max(l, s) for l, s in zip(lstm_scores, sarima_scores)]
lstm_normalized = [l / m for l, m in zip(lstm_scores, max_vals)]
sarima_normalized = [s / m for s, m in zip(sarima_scores, max_vals)]

# Prepare for radar plotting
labels = np.array(metrics)
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

lstm_normalized += lstm_normalized[:1]
sarima_normalized += sarima_normalized[:1]

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
ax.plot(angles, lstm_normalized, label="LSTM", marker='o', linewidth=2)
ax.fill(angles, lstm_normalized, alpha=0.25)

ax.plot(angles, sarima_normalized, label="SARIMA", marker='x', linewidth=2)
ax.fill(angles, sarima_normalized, alpha=0.25)

ax.set_title("LSTM vs SARIMA – Normalized Forecast Metrics", size=14)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticklabels([])
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()

radar_chart_path = "artifacts/lstm_vs_sarima_radar.png"
plt.savefig(radar_chart_path)
plt.close()

print(f"Radar chart saved to: {radar_chart_path}")
