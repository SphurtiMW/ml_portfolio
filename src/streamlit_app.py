import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Constants
MODEL_PATH = "checkpoints/best_model.keras"
DATA_PATH = "artifacts/Historical Product Demand.csv"
SEQ_LENGTH = 30

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date", low_memory=False)
    df = df[['Order_Demand']]
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df['Order_Demand_Log'] = np.log1p(df['Order_Demand'])
    return df

def prepare_input(df, scaler):
    df['MA_7'] = df['Order_Demand_Log'].rolling(7).mean()
    df['STD_7'] = df['Order_Demand_Log'].rolling(7).std()
    df['Lag_1'] = df['Order_Demand_Log'].shift(1)
    df['DayOfWeek'] = df.index.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Month'] = df.index.month
    df.dropna(inplace=True)
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns), scaler

def create_input_sequence(df_scaled):
    last_window = df_scaled[-SEQ_LENGTH:].copy()
    return last_window.values.reshape(1, SEQ_LENGTH, -1)

# UI
st.title(" Demand Forecasting App")
model = load_model()
raw_df = load_data()
scaler = MinMaxScaler()

df_scaled, scaler = prepare_input(raw_df.copy(), scaler)
input_seq = create_input_sequence(df_scaled)

if st.button("Predict Demand"):
    pred_log = model.predict(input_seq)[0][0]
    pred_demand = np.expm1(pred_log)
    st.success(f"Predicted Next-Day Demand: **{pred_demand:.2f} units**")

    # Optional: plot last 30 days
    st.subheader("Last 30 Days (Log Transformed)")
    st.line_chart(raw_df['Order_Demand_Log'].tail(SEQ_LENGTH))
