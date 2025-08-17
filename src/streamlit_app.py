import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- Constants ---
MODEL_PATH = "checkpoints/optimized_lstm_model.keras"
DATA_PATH = "artifacts/Historical Product Demand.csv"
SEQ_LENGTH = 30
FUTURE_DAYS = 14
TARGET_COL = "Order_Demand_Log"

# --- App Setup ---
st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("Supply Chain Demand Forecasting App")
st.markdown("Predict future demand using an LSTM model trained on historical order data.")

# --- Load model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date", low_memory=False)
    df = df[['Order_Demand']]
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df[TARGET_COL] = np.log1p(df['Order_Demand'])
    return df

# --- Feature engineering + scaling ---
def prepare_input(df, scaler):
    df['MA_7'] = df[TARGET_COL].rolling(7).mean()
    df['STD_7'] = df[TARGET_COL].rolling(7).std()
    df['Lag_1'] = df[TARGET_COL].shift(1)
    df['DayOfWeek'] = df.index.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Month'] = df.index.month
    df.dropna(inplace=True)
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns), scaler

# --- Create sequence input for LSTM ---
def create_input_sequence(df_scaled):
    last_window = df_scaled[-SEQ_LENGTH:].copy()
    return last_window.values.reshape(1, SEQ_LENGTH, -1)

# --- Multi-step future forecast ---
def forecast_future(df_scaled, model, days=FUTURE_DAYS):
    preds = []
    current_input = df_scaled[-SEQ_LENGTH:].copy()

    for _ in range(days):
        input_array = current_input.values.reshape(1, SEQ_LENGTH, -1)
        next_pred = model.predict(input_array, verbose=0)[0][0]
        next_row = current_input.iloc[-1].copy()
        next_row[TARGET_COL] = next_pred
        next_index = current_input.index[-1] + pd.Timedelta(days=1)
        current_input = pd.concat([
            current_input.iloc[1:], 
            pd.DataFrame([next_row], index=[next_index])
        ])
        preds.append((next_index, np.expm1(next_pred)))

    forecast_df = pd.DataFrame(preds, columns=["Date", "Predicted Demand"]).set_index("Date")
    return forecast_df

# --- Run the App ---
model = load_model()
raw_df = load_data()
scaler = MinMaxScaler()
df_scaled, scaler = prepare_input(raw_df.copy(), scaler)
input_seq = create_input_sequence(df_scaled)

if st.button(" Predict Next-Day Demand"):
    pred_log = model.predict(input_seq, verbose=0)[0][0]
    pred_demand = np.expm1(pred_log)
    st.success(f" Predicted Next-Day Demand: **{pred_demand:.2f} units**")

    st.subheader("Last 30 Days (Log Transformed)")
    st.line_chart(raw_df[TARGET_COL].tail(SEQ_LENGTH))

if st.button("Forecast Next 14 Days"):
    forecast_df = forecast_future(df_scaled, model)
    st.subheader("Future Forecast (14 Days)")
    st.line_chart(forecast_df["Predicted Demand"])

    st.download_button(
        label="Download Forecast CSV",
        data=forecast_df.to_csv().encode("utf-8"),
        file_name="future_forecast.csv",
        mime="text/csv"
    )
