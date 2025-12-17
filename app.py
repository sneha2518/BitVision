import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="BitVision", layout="centered")

st.title("BitVision – Bitcoin Price Prediction")
st.write("Enter the previous day's Bitcoin Close price to predict the next day's price.")

# ------------------------------
# LOAD MODEL
# ------------------------------
model = load_model("bitcoin_lstm_final_model.h5")

LOOKBACK = 30
FEATURES = 12   # must match training

# ------------------------------
# USER INPUT
# ------------------------------
prev_close = st.number_input(
    "Enter Previous Day Bitcoin Close Price (USD)",
    min_value=0.0,
    value=30000.0,
    step=100.0
)

# ------------------------------
# PREDICTION
# ------------------------------
if st.button("Predict Next Day Price"):

    # Create dummy dataframe (same structure as training)
    data = {
        "Open": [prev_close] * LOOKBACK,
        "High": [prev_close] * LOOKBACK,
        "Low": [prev_close] * LOOKBACK,
        "Close": [prev_close] * LOOKBACK,
        "Volume": [0] * LOOKBACK,
        "Return": [0] * LOOKBACK,
        "LogReturn": [0] * LOOKBACK,
        "MA_7": [prev_close] * LOOKBACK,
        "MA_30": [prev_close] * LOOKBACK,
        "Volatility_7": [0] * LOOKBACK,
        "RSI": [50] * LOOKBACK,
        "MACD": [0] * LOOKBACK,
    }

    df = pd.DataFrame(data)

    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    # Reshape for LSTM
    X = scaled.reshape(1, LOOKBACK, FEATURES)

    # Predict
    prediction_scaled = model.predict(X)

    # Inverse scale (only Close column)
    dummy = np.zeros((1, FEATURES))
    dummy[0, 3] = prediction_scaled[0][0]  # Close index = 3

    predicted_close_usd = scaler.inverse_transform(dummy)[0][3]

    # USD → INR conversion
    USD_TO_INR = 83.0
    predicted_close_inr = predicted_close_usd * USD_TO_INR

    # ------------------------------
    # OUTPUT
    # ------------------------------
    st.subheader("Prediction Result")
    st.success(f"Predicted Next Day Bitcoin Price: ₹ {predicted_close_inr:,.2f}")

    st.info("Prediction is based on historical patterns learned by the LSTM model.")
    

