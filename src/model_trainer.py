import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Config
SEQ_LENGTH = 30
SPLIT_DAYS = 90
FUTURE_DAYS = 14
TARGET_COL = 'Order_Demand_Log'

# Optimize CPU
os.environ["OMP_NUM_THREADS"] = "4"
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# --- Load & Prepare ---
df = pd.read_csv(
    r"C:\Users\sphur\Downloads\ml_portfolio-main\artifacts\Historical Product Demand.csv",
    parse_dates=["Date"], index_col="Date", low_memory=False
)[['Order_Demand']]

df = df[~df.index.duplicated(keep='first')].sort_index()

# --- Features ---
df[TARGET_COL] = np.log1p(df['Order_Demand'])
df['MA_7'] = df[TARGET_COL].rolling(7).mean()
df['STD_7'] = df[TARGET_COL].rolling(7).std()
df['Lag_1'] = df[TARGET_COL].shift(1)
df['DayOfWeek'] = df.index.dayofweek
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['Month'] = df.index.month
df.dropna(inplace=True)

# --- VIF Filtering ---
features = df.columns.tolist()
features.remove('Order_Demand')
X_vif = df[features]
vif_df = pd.DataFrame({
    'feature': features,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(len(features))]
})
print(vif_df)

to_drop = vif_df[vif_df['VIF'] > 15]['feature'].tolist()
to_drop = [f for f in to_drop if f not in [TARGET_COL, 'IsWeekend', 'Month']]
df.drop(columns=to_drop, inplace=True)

# --- Scaling ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)
df_scaled.to_hdf('processed_data.h5', key='df', mode='w')

# --- Train/Test Split ---
test_start = df_scaled.index.max() - pd.Timedelta(days=SPLIT_DAYS)
train_df = df_scaled[df_scaled.index <= test_start]
test_df = df_scaled[df_scaled.index > test_start]

# --- Sequence Generation ---
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)
        y.append(data.iloc[i + seq_length][TARGET_COL])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_df, SEQ_LENGTH)
X_test, y_test = create_sequences(test_df, SEQ_LENGTH)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
if len(X_test) == 0 or len(X_train) == 0:
    raise ValueError("Insufficient data after sequence creation.")

# --- Model Architecture ---
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, X_train.shape[2])),
    Dropout(0.1),
    LSTM(64),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# --- Callbacks ---
checkpoint_path = "checkpoints/best_model.h5"
os.makedirs("checkpoints", exist_ok=True)

callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# --- Train ---
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# --- Predict & Evaluate ---
y_pred = model.predict(X_test).flatten()
y_pred_inv = np.expm1(y_pred)
y_test_inv = np.expm1(y_test)

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")

# --- Plot Test Set ---
plt.figure(figsize=(12, 6))
plt.plot(test_df.index[SEQ_LENGTH:], y_test_inv, label="Actual", color="blue")
plt.plot(test_df.index[SEQ_LENGTH:], y_pred_inv, label="Predicted", color="red")
plt.title("Demand Forecasting - Test Set")
plt.xlabel("Date")
plt.ylabel("Order Demand")
plt.legend()
plt.show()

# --- Future Forecast ---
last_window = df_scaled[-SEQ_LENGTH:].copy()
future_preds = []
current_input = last_window.copy()

for _ in range(FUTURE_DAYS):
    input_array = current_input.values.reshape(1, SEQ_LENGTH, -1)
    next_pred = model.predict(input_array)[0][0]
    next_row = current_input.iloc[-1].copy()
    next_row[TARGET_COL] = next_pred
    current_input = pd.concat([
        current_input.iloc[1:],
        pd.DataFrame([next_row], index=[current_input.index[-1] + pd.Timedelta(days=1)])
    ])
    future_preds.append(np.expm1(next_pred))

# --- Plot Future Forecast ---
future_dates = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=FUTURE_DAYS)
plt.figure(figsize=(12, 6))
plt.plot(future_dates, future_preds, marker='o', color='green')
plt.title("Future Forecast (Next 14 Days)")
plt.xlabel("Date")
plt.ylabel("Predicted Order Demand")
plt.grid(True)
plt.show()

# --- Save Final Model ---
model.save("optimized_lstm_model.keras")
print("Model saved in .keras format successfully!")



"""""os.environ["OMP_NUM_THREADS"] = "4"
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
print("Model saved successfully!")"""""
