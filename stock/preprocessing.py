import streamlit as st
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
import io

st.set_page_config(layout='wide')
st.title('📊 Data Preprocessing')

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

# Load data
df = pd.read_csv(CURRENT_DIR / 'StockData.csv')

# Ensure that the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

st.subheader('📈 Stock Closing Prices Over Time')
st.line_chart(df, x='Date', y='Close', y_label='Close price')

st.divider()
st.subheader('🔬 Importance of Understanding Trend & Seasonality for Forecasting')
st.write('''
         1. **Model Selection**
            - If the trend is nonlinear, models like ARIMA with a trend term, Prophet, or LSTMs might be suitable.
            - If the seasonal component is strong, SARIMA, seasonal decomposition methods, or Fourier transformations in regression models could be useful.
         2. **Feature Engineering**
            - Removing the trend before applying certain models (like ARIMA) can improve forecasting accuracy.
            - Seasonal patterns should be incorporated into the model if they persist over time.
         3. **Structural Changes**
            - The sharp trend shift suggests that a piecewise regression model or a change-point detection method might be necessary to handle different behaviors before and after the shift.
         ''')

st.divider()

df.set_index('Date', inplace=True)  # Set 'Date' as index

# Perform seasonal decomposition
decomposition = seasonal_decompose(df['Close'], model='additive', period=30)  # Assuming monthly seasonality

st.subheader('📈 Decompose the time series into trend')
st.line_chart(pd.DataFrame(decomposition.trend))
st.write('''
         - The trend shows a gradual increase from 2023 to early 2025, followed by a sharp upward shift around early 2025.
         - After the shift, the trend continues to increase but at a steadier rate.
         - This suggests a structural change in the data, possibly due to an external event, market shift, or policy change.''')
st.divider()

st.subheader('📉 Decompose the time series into seasonality')
st.line_chart(pd.DataFrame(decomposition.seasonal))
st.write('''
         - The seasonal component exhibits periodic fluctuations that repeat consistently over time.
         - The amplitude of seasonality remains relatively stable before and after the trend shift, meaning the seasonal effects are not changing in magnitude.''')
st.divider()

st.subheader('📈 Decompose the time series into Residual')
st.line_chart(pd.DataFrame(decomposition.resid))

st.subheader('🔍 Restructure the time series using a sliding window approach and Split the Closing prices time series into 80% for training and 20% for testing')

with st.echo():
    # Assume df contains a "Closing Price" column
    closing_prices = df["Close"].values

    # Normalize data
    scaler = MinMaxScaler()
    closing_prices = scaler.fit_transform(closing_prices.reshape(-1, 1)).flatten()

    # Sliding window transformation
    window_size = 10
    X, y = [], []

    for i in range(len(closing_prices) - window_size):
        X.append(closing_prices[i:i + window_size])  # Features
        y.append(closing_prices[i + window_size])    # Target

    X = np.array(X)
    y = np.array(y)

    # Split into training (80%) and testing (20%)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

with st.container(border=True):
    st.write(f"X_train shape: {X_train.shape}")
    st.write(f"X_test shape: {X_test.shape}")
    st.write(f"y_train shape: {y_train.shape}")
    st.write(f"y_test shape: {y_test.shape}")

st.divider()

st.subheader('📚 Original dataset with all features')
st.dataframe(df)