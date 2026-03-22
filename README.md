# 📈 BitVision – Bitcoin Price Prediction using LSTM

BitVision is a deep learning–based web application that predicts the **next day’s Bitcoin closing price (in INR)** using a trained **Long Short-Term Memory (LSTM)** model.

This project demonstrates an end-to-end pipeline for **financial time-series forecasting**, including data preprocessing, feature engineering, model training, and deployment using **Streamlit**.

---

## 🚀 Live Demo

https://bitvision.streamlit.app/

---

## 📌 Project Overview

- **Domain:** Deep Learning, Financial Time-Series Analysis  
- **Model Type:** LSTM (Regression)  
- **Frameworks:** TensorFlow, Keras, Streamlit  
- **Prediction Target:** Next-day Bitcoin closing price  
- **Output Currency:** INR (₹)  

The LSTM model captures temporal dependencies in historical Bitcoin data to generate accurate future predictions.

---

## 🧠 Features

- 📊 Time-series prediction using LSTM  
- 🔮 Predicts next-day Bitcoin price  
- ⚡ Real-time user input prediction  
- 🌐 Interactive Streamlit interface  
- 📉 Uses multiple financial indicators  

---

## ⚙️ Application Workflow

1. User enters today's Bitcoin closing price  
2. System loads recent historical data  
3. A 30-day sequence is created  
4. Data is scaled using trained scalers  
5. LSTM model predicts next-day price  
6. Result is displayed in INR  

---

## 🧠 Model Architecture

- LSTM (64 units, return_sequences=True)  
- Dropout (0.2)  
- LSTM (64 units)  
- Dropout (0.2)  
- Dense (32 units, ReLU)  
- Output Layer (1 neuron)  

**Loss Function:** Mean Squared Error (MSE)  
**Optimizer:** Adam  
**Lookback Window:** 30 days  

---

## 📊 Feature Engineering

The model uses the following features:

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

All features are normalized using **MinMaxScaler**, and the same scaler is reused during deployment.

---

## 🧠 Deployment Logic

Since LSTM requires sequential input:

1. Recent historical data is loaded internally  
2. User input is appended  
3. A 30-day sequence is constructed  
4. Model predicts next-day price  

This ensures both:
✔ Accurate predictions  
✔ Simple user experience  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- Pandas & NumPy  
- Scikit-learn  
- Streamlit  

---

## 📂 Project Structure

bitvision/
│── app.py  
│── model.h5  
│── scaler.pkl  
│── dataset/  
│── notebook.ipynb  
│── README.md  

---

## 💻 How to Run Locally

1. Clone the repository:
git clone https://github.com/sneha2518/bitvision.git

2. Navigate to the folder:
cd bitvision

3. Install dependencies:
pip install -r requirements.txt

4. Run the application:
streamlit run app.py

---

## 🌐 Deployment

The application is deployed using **Streamlit Cloud** and is accessible via a web interface.

---

## 💡 Future Improvements

- Improve model accuracy with advanced architectures  
- Add real-time API data integration  
- Enhance UI/UX design  
- Support multiple cryptocurrencies  

---

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used for financial decision-making.

---

## 👩‍💻 Author

Sneha Eppanapally

---
