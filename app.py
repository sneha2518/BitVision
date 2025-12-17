import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model & scaler
model = load_model("bitcoin_lstm_final_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("Bitcoin Price Prediction")

latest_price = st.number_input(
    "Enter Latest Bitcoin Price",
    min_value=0.0,
    format="%.2f"
)

if st.button("Predict"):
    # Convert input to numpy
    input_data = np.array([[latest_price]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Reshape for LSTM (samples, timesteps, features)
    input_scaled = input_scaled.reshape(1, 1, 1)

    # Predict
    prediction_scaled = model.predict(input_scaled)

    # Inverse scaling
    prediction = scaler.inverse_transform(prediction_scaled)

    st.success(f"Predicted Bitcoin Price: {prediction[0][0]:.2f}")


