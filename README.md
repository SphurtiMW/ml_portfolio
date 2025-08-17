# Supply Chain Demand Forecasting

## Supply Chain Demand Forecasting using LSTM and SARIMA

A predictive analytics project that leverages deep learning (LSTM) and classical time series modeling (SARIMA) to forecast future product demand.

---

## 🚀 Introduction

In modern supply chains, accurate demand forecasting is critical to minimizing inventory costs, improving customer satisfaction, and optimizing production planning. Traditional methods often fall short when faced with nonlinear patterns, high seasonality, and long-term dependencies in data. This project addresses the **problem of short-term demand forecasting** by applying and comparing two powerful time series forecasting models:

- **LSTM (Long Short-Term Memory)**: A deep learning model capable of learning from sequences and remembering long-term dependencies. It is well-suited for noisy and nonlinear time series data.
- **SARIMA (Seasonal AutoRegressive Integrated Moving Average)**: A classical statistical model that performs well with seasonal and stationary time series data.

The main objective is to forecast **next-day product demand** using historical data, and visualize comparative model performance to help stakeholders make informed inventory decisions.

---

## 🧠 Models Used & Why

### 1. LSTM (Long Short-Term Memory)
- Captures long-range dependencies and trends in sequences.
- More robust to irregular patterns and missing values.
- Architecture includes stacked LSTM layers followed by dense layers for regression.

### 2. SARIMA (Seasonal ARIMA)
- Incorporates seasonality and trend using traditional time series decomposition.
- Simpler and more interpretable.
- Effective when the time series has consistent seasonal patterns.

### Why Use Both?
Comparing deep learning and classical models provides more transparency in results. It also allows decision-makers to choose the right model based on forecast horizon, interpretability, or computational cost.

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

Optional: GitHub Actions CI/CD for automation 

---

## Live Demo

> Currently under deployment – Link coming soon

---

## Future Improvements:

 Add holidays/seasonality features

 Multi-product or region-based forecasting

 Scheduled model retraining

 Forecast uncertainty bands