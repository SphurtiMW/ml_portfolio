import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

os.environ["OMP_NUM_THREADS"] = "4"
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

df = pd.read_csv("Historical Product Demand.csv", parse_dates=["Date"], index_col="Date")
df = df[['Order_Demand']]  

df['Order_Demand_Log'] = np.log1p(df['Order_Demand'])

scaler = MinMaxScaler(feature_range=(0, 1))
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]


scaler.fit(train_data[['Order_Demand_Log']])
df_scaled = scaler.transform(df[['Order_Demand_Log']])


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])  
    return np.array(X), np.array(y)

SEQ_LENGTH = 60  
X, y = create_sequences(df_scaled, SEQ_LENGTH)

X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),  
    Dropout(0.2),
    LSTM(32, return_sequences=False), 
    Dropout(0.2),
    Dense(16, activation="relu"),  
    Dense(1) 
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=1)

y_pred = model.predict(X_test)

y_pred_actual = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label="Actual Demand", color="blue")
plt.plot(y_pred_actual, label="Predicted Demand", color="red")
plt.title("LSTM Demand Forecasting (Optimized for CPU)")
plt.xlabel("Time")
plt.ylabel("Order Demand")
plt.legend()
plt.show()

model.save("optimized_lstm_model.h5")
print("Model saved successfully!")
