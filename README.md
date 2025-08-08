# Supply Chain Demand Forecasting

A Streamlit-based web application that predicts future product demand using an LSTM model trained on historical order data. Built with TensorFlow, pandas, and deployed for real-time forecasting.

---

## Live Demo

> Currently under deployment – Link coming soon

---

## Features

- Predict next-day product demand
- Extend forecasting up to 14 days
- Visualize historical trends (log-transformed)
- Export charts and forecast results
- Lightweight and cloud-deployable

---

## Model Architecture

Input → LSTM(128, return_sequences=True)
→ Dropout(0.1)
→ LSTM(64)
→ Dropout(0.1)
→ Dense(16, activation='relu')
→ Dense(1)


- LSTM layers capture time dependencies in sequences
- Dropout layers reduce overfitting
- Dense layers refine to a single demand output

---

## Data & Preprocessing

- Dataset: Historical daily order demand
- Target Variable: Log-transformed `Order_Demand`
- Features:
  - `MA_7` – 7-day moving average
  - `STD_7` – 7-day rolling standard deviation
  - `Lag_1` – Previous day's demand
  - `DayOfWeek`, `IsWeekend`, `Month`
- Scaling: `MinMaxScaler`
- Sequence Length: 30 time steps

---

## Model Performance

| Metric | Value |
|--------|-------|
| MAE    | 0.09  |
| RMSE   | 0.19  |

- MAE (Mean Absolute Error): Measures average absolute error
- RMSE (Root Mean Squared Error): Penalizes large deviations

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/ml_portfolio.git
cd ml_portfolio

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run src/streamlit_app.py

ml_portfolio/
├── artifacts/
│   ├── Historical Product Demand.csv
│   └── test_forecast.png (optional)
├── checkpoints/
│   └── best_model.keras
├── src/
│   ├── model_trainer.py
│   ├── evaluations.py
│   └── streamlit_app.py
├── requirements.txt
├── Dockerfile
└── README.md

Deployment
Platform: Streamlit Cloud or Hugging Face Spaces

Docker support for containerized deployment

Optional: GitHub Actions CI/CD for automation '''

Future Improvements:

 Add holidays/seasonality features

 Multi-product or region-based forecasting

 Scheduled model retraining

 Forecast uncertainty bands