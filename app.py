import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="BitVision", layout="centered")

st.title("BitVision – Bitcoin Price Prediction")
st.write(
    "Enter **today’s Bitcoin Close price** to predict **tomorrow’s price** "
    "using an LSTM deep learning model."
)

st.caption(
    "Note: LSTM models require historical context. "
    "The application uses recent historical data internally."
)

# --------------------------------------------------
# LOAD MODEL & SCALERS
# --------------------------------------------------
model = load_model("bitcoin_lstm_final_model.h5")
feature_scaler = joblib.load("feature_scaler.pkl")
close_scaler = joblib.load("close_scaler.pkl")

LOOKBACK = 30

# --------------------------------------------------
# LOAD INTERNAL HISTORICAL DATA
# --------------------------------------------------
history_df = pd.read_csv("history.csv")

history_df = history_df[['Open', 'High', 'Low', 'Close', 'Volume']]

# --------------------------------------------------
# FEATURE ENGINEERING (MATCH TRAINING)
# --------------------------------------------------
history_df['Return'] = history_df['Close'].pct_change()
history_df['LogReturn'] = np.log(history_df['Close'] / history_df['Close'].shift(1))

history_df['MA_7'] = history_df['Close'].rolling(7).mean()
history_df['MA_30'] = history_df['Close'].rolling(30).mean()

history_df['Volatility_7'] = history_df['Return'].rolling(7).std()
history_df['Close_lag1'] = history_df['Close'].shift(1)

delta = history_df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-9)
history_df['RSI'] = 100 - (100 / (1 + rs))

ema12 = history_df['Close'].ewm(span=12, adjust=False).mean()
ema26 = history_df['Close'].ewm(span=26, adjust=False).mean()
history_df['MACD'] = ema12 - ema26

history_df = history_df.dropna()

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
today_price = st.number_input(
    "Enter today’s Bitcoin Close price (USD)",
    min_value=0.0,
    format="%.2f"
)

# --------------------------------------------------
# PREDICTION LOGIC
# --------------------------------------------------
if today_price > 0:

  # last 29 historical rows
last_29 = history_df.iloc[-29:].copy()

# create today's row
new_row = last_29.iloc[-1].copy()
new_row["Close"] = today_price
new_row["Close_lag1"] = last_29.iloc[-1]["Close"]

# combine to make 30-day sequence
sequence_df = pd.concat([last_29, new_row.to_frame().T])

    # scale
    sequence_scaled = feature_scaler.transform(sequence_df)

    # reshape for LSTM
    X = sequence_scaled.reshape(1, LOOKBACK, sequence_scaled.shape[1])

    # predict
    predicted_scaled = model.predict(X)[0][0]

    # inverse scale
    predicted_price = close_scaler.inverse_transform(
        [[predicted_scaled]]
    )[0][0]

    # output
    st.subheader("Prediction Result")
    USD_TO_INR = 83.0  # approx conversion rate

predicted_price_inr = predicted_price * USD_TO_INR

st.success(
    f"📈 Predicted Bitcoin Close Price for Tomorrow: "
    f"**₹{predicted_price_inr:,.2f} INR**"
)


else:
    st.info("Please enter today’s Bitcoin price to get tomorrow’s prediction.")


