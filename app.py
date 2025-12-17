import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="BitVision – Bitcoin Price Prediction",
    layout="centered"
)

# --------------------------------------------------
# TITLE & DESCRIPTION
# --------------------------------------------------
st.title("BitVision – Bitcoin Price Prediction using LSTM")

st.write(
    """
    This web application predicts the **next Bitcoin closing price**
    using a **Long Short-Term Memory (LSTM)** deep learning model.
    Upload a Bitcoin OHLCV CSV file to get the prediction.
    """
)

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
    # BASE COLUMNS (EXACT)
    # -----------------------------
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # -----------------------------
    # FEATURE ENGINEERING (MATCH TRAINING)
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

    # -----------------------------
    # PREDICTION
    # -----------------------------
    prediction = model.predict(X)
    predicted_scaled = prediction[-1][0]

    # -----------------------------
    # INVERSE SCALING (CLOSE PRICE)
    # -----------------------------
    close_index = df.columns.get_loc("Close")

    dummy = np.zeros((1, scaled_data.shape[1]))
    dummy[0, close_index] = predicted_scaled

    inversed = scaler.inverse_transform(dummy)
    predicted_close_usd = inversed[0, close_index]

    # -----------------------------
    # USD → INR CONVERSION
    # -----------------------------
    USD_TO_INR = 83.0  # approximate exchange rate
    predicted_close_inr = predicted_close_usd * USD_TO_INR

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Prediction Result")

    st.success(
        f"Predicted Next Bitcoin Close Price: ₹ {predicted_close_inr:,.2f} INR"
    )

    st.caption(
        "Price is inverse-scaled and converted from USD to INR for better understanding."
    )
