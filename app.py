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
st.write("Upload Bitcoin OHLCV CSV file to predict the next Close price.")

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
model = load_model("bitcoin_lstm_final_model.h5")

# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Bitcoin CSV file (Open, High, Low, Close, Volume)",
    type=["csv"]
)

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded_file is not None:

    # -----------------------------
    # READ DATA
    # -----------------------------
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # BASE COLUMNS
    # -----------------------------
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # -----------------------------
    # FEATURE ENGINEERING (EXACT MATCH)
    # -----------------------------
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

    # -----------------------------
    # CLEAN DATA
    # -----------------------------
    df = df.dropna()

    # -----------------------------
    # SCALING
    # -----------------------------
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # -----------------------------
    # CREATE SEQUENCES
    # -----------------------------
    LOOKBACK = 30
    X = []

    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i - LOOKBACK:i])

    X = np.array(X, dtype="float32")

    # Debug shape (important)
    st.write("Input shape to model:", X.shape)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    prediction = model.predict(X)
    predicted_value = prediction[-1][0]

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Prediction Result")
    st.success(
        f"Predicted Next Bitcoin Close Price (scaled value): {predicted_value:.4f}"
    )

    st.info(
        "Prediction is shown on scaled values as per the training pipeline."
    )


