import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Demand Forecast Dashboard", layout="centered")

st.title("Product Demand Forecast")

# Load model
MODEL_PATH = "models/optimized_lstm_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load sample data for prediction
sample_data = pd.read_csv("artifacts/Historical Product Demand.csv", parse_dates=["Date"])
st.write("### Sample Data", sample_data.tail())

# Display prediction on the latest sequence
if st.button("Run Prediction on Latest Sequence"):
    # Preprocessing (simplified - adapt if needed)
    recent = sample_data["Order_Demand"].tail(60).values
    recent = np.log1p(recent)
    recent = (recent - recent.min()) / (recent.max() - recent.min())  # dummy scaling
    recent = np.expand_dims(recent, axis=(0, -1))  # shape (1, 60, 1)

    pred = model.predict(recent)[0][0]
    st.success(f"🔮 Predicted (scaled log): {pred:.4f}")
