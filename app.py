import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG (must be first Streamlit command)
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
    "This application predicts the **next Bitcoin closing price** using a "
    "deep learning **LSTM model** trained on historical OHLCV data."
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
model = load_model("bitcoin_lstm_final_model.h5")
st.success("LSTM model loaded successfully")

# Show LSTM architecture
with st.expander("View LSTM Model Architecture"):
    model.summary(print_fn=lambda x: st.text(x))

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
    # FEATURE ENGINEERING (EXACT MATCH WITH TRAINING)
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

    st.write("Input shape to LSTM:", X.shape)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    st.write("Running LSTM prediction...")
    prediction_scaled = model.predict(X)

    # -----------------------------
    # INVERSE SCALING (ONLY CLOSE PRICE)
    # -----------------------------
    close_index = df.columns.get_loc("Close")

    dummy = np.zeros((1, df.shape[1]))
    dummy[0, close_index] = prediction_scaled[-1][0]

    predicted_close_usd = scaler.inverse_transform(dummy)[0, close_index]

    # -----------------------------
    # USD → INR CONVERSION
    # -----------------------------
    USD_TO_INR = 83.0  # approximate current rate
    predicted_close_inr = predicted_close_usd * USD_TO_INR

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Prediction Result")

    st.success(
        f"Predicted Next Bitcoin Close Price:\n\n"
        f"💵 **USD:** ${predicted_close_usd:,.2f}\n\n"
        f"🇮🇳 **INR:** ₹{predicted_close_inr:,.2f}"
    )

    st.info(
        "The prediction is generated using an LSTM deep learning model "
        "and converted to Indian Rupees for user convenience."
    )

