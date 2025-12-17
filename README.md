# BitVision – Bitcoin Price Prediction using LSTM

BitVision is a deep learning–based web application that predicts the **next day’s Bitcoin closing price (in INR)** using a trained **Long Short-Term Memory (LSTM)** network.

The project demonstrates an end-to-end implementation of **financial time-series forecasting**, covering data preprocessing, feature engineering, model training, and real-world **deployment using Streamlit**.

---

## Project Overview

- **Domain:** Deep Learning, Financial Time-Series Analysis  
- **Model Type:** LSTM (Regression)  
- **Frameworks:** TensorFlow, Keras, Streamlit  
- **Prediction Target:** Next-day Bitcoin Close price  
- **Output Currency:** INR (Indian Rupees)

The LSTM model captures temporal dependencies in historical Bitcoin market data to forecast future prices.

---

## Application Functionality

- User enters **today’s Bitcoin Close price**
- The application internally uses **recent historical market data**
- A **30-day time-series input sequence** is constructed
- The trained **LSTM model** predicts **tomorrow’s price**
- The predicted value is displayed in **INR**

> Although the user provides only a single input, historical context is maintained internally to satisfy LSTM requirements.

---

## Model Architecture

- LSTM layer (64 units, return sequences = True)
- Dropout (0.2)
- LSTM layer (64 units)
- Dropout (0.2)
- Dense layer (32 units, ReLU)
- Output layer (1 neuron)

**Loss Function:** Mean Squared Error (MSE)  
**Optimizer:** Adam  
**Lookback Window:** 30 days  

---

## Feature Engineering

The model was trained using the following features:

- Open  
- High  
- Low  
- Close  
- Volume  
- Return  
- Log Return  
- Moving Averages (7-day, 30-day)  
- Volatility  
- Lagged Close  
- RSI  
- MACD  

All features are normalized using **MinMaxScaler**.  
The same scalers used during training are reused during deployment to ensure consistency.

---

## Deployment Logic

LSTM models require temporal context.  
To enable a simple user interface while maintaining model correctness, the application:

1. Loads recent historical data internally  
2. Appends today’s user-provided price  
3. Constructs a 30-day input sequence  
4. Predicts the next day’s Bitcoin closing price  

This approach ensures both **academic validity** and **practical usability**.

---




