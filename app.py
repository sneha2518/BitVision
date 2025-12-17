import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="BitVision", layout="centered")

st.title("BitVision – Bitcoin Price Prediction using LSTM")
st.write(
    "This application predicts the next Bitcoin closing price using a trained LSTM model."
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = load_model("bitcoin_lstm_final_model.h5")

# --------------------------------------------------
# LOAD DATA INTERNALLY (NO USER UPLOAD)
# --------------------------------------------------
df = pd.read_csv("bitcoin_small.csv")

# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# --------------------------------------------------
# FEATURE ENGINEERING (SAME AS TRAINING)
# --------------------------------------------------
df['Return'] = df['Close'].pct_change()
df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))

df['MA_7'] = df['Close'].rolling(7).mean()
df['MA_30'] = df['Close'].rolling(30).mean()

df['Volatility_7'] = df['Return'].rolling(7).std()
df['Close_lag1'] = df['Close'].shift(1)

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
RS = gain / (loss + 1e-9)
df['RSI'] = 100 - (100 / (1 + RS))

# MACD
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26

# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------
df = df.dropna()

# --------------------------------------------------
# SCALING
# --------------------------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# --------------------------------------------------
# CREATE LSTM SEQUENCES
# --------------------------------------------------
LOOKBACK = 30
X = []

for i in range(LOOKBACK, len(scaled_data)):
    X.append(scaled_data[i - LOOKBACK:i])

X = np.array(X, dtype="float32")

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------
if st.button("Predict Next Bitcoin Price"):

    prediction_scaled = model.predict(X)
    last_scaled_value = prediction_scaled[-1][0]

    # Inverse scaling (Close price only)
    close_index = df.columns.get_loc("Close")
    dummy = np.zeros((1, scaled_data.shape[1]))
    dummy[0, close_index] = last_scaled_value
    predicted_close_usd = scaler.inverse_transform(dummy)[0, close_index]

    # USD → INR
    USD_TO_INR = 83.0
    predicted_close_inr = predicted_close_usd * USD_TO_INR

    st.success(f"Predicted Next Bitcoin Close Price: ₹ {predicted_close_inr:,.2f} INR")

    st.caption(
        "Prediction is based on historical time-series data using an LSTM deep learning model."
    )
