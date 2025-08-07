import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# --- CONFIG ---
MODEL_PATH = "checkpoints/best_model.h5"
DATA_PATH = "artifacts/Historical Product Demand.csv"
SEQ_LENGTH = 30
FUTURE_DAYS = 14
TARGET_COL = 'Order_Demand_Log'

st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date", low_memory=False)
    df = df[['Order_Demand']].sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df[TARGET_COL] = np.log1p(df['Order_Demand'])
    df['MA_7'] = df[TARGET_COL].rolling(7).mean()
    df['STD_7'] = df[TARGET_COL].rolling(7).std()
    df['Lag_1'] = df[TARGET_COL].shift(1)
    df['DayOfWeek'] = df.index.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Month'] = df.index.month
    df.dropna(inplace=True)
    return df

df = load_data()

# --- Scale ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- Forecast ---
def forecast_future(df_scaled, days=FUTURE_DAYS):
    window = df_scaled[-SEQ_LENGTH:].copy()
    predictions = []
    input_seq = window.copy()

    for _ in range(days):
        input_array = input_seq.values.reshape(1, SEQ_LENGTH, -1)
        pred = model.predict(input_array)[0][0]
        new_row = input_seq.iloc[-1].copy()
        new_row[TARGET_COL] = pred
        input_seq = pd.concat([
            input_seq.iloc[1:],
            pd.DataFrame([new_row], index=[input_seq.index[-1] + pd.Timedelta(days=1)])
        ])
        predictions.append(np.expm1(pred))
    
    future_dates = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=days)
    return pd.Series(predictions, index=future_dates)

# --- UI ---
st.title(" Demand Forecasting Dashboard")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Historical Demand")
    st.line_chart(df['Order_Demand'])

with col2:
    st.subheader("🔮 Future Forecast (Next 14 Days)")
    forecast = forecast_future(df_scaled)
    st.line_chart(forecast)

st.markdown("---")
st.subheader("Model Info")
st.write(f"Model path: `{MODEL_PATH}`")
st.write(f"Data path: `{DATA_PATH}`")
st.write(f"Forecasting window: `{SEQ_LENGTH}` days")
