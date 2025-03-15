import streamlit as st
import pandas as pd
import pathlib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from IPython.display import Markdown

st.set_page_config(layout='wide')
st.title('✈️ 1D Convolutional Neural Network (CNN)')

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
df = pd.read_csv(CURRENT_DIR / 'StockData.csv')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    layer = tf.keras.layers.Dense(64, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))
    
set_seed(42)

st.subheader('❓ Why Use 1D CNN for times series forecasting?')
st.write('''A 1D Convolutional Neural Network (CNN) is commonly used for time series forecasting because it is effective at capturing temporal patterns and local dependencies in the data.
The convolutional layers work by applying filters (kernels) to local segments of the input data, allowing the model to learn relevant features in a hierarchical manner.

The key reasons for using a 1D CNN for time series forecasting are:
- Local Feature Extraction: Time series data often contain local patterns or trends (e.g., seasonality or short-term fluctuations), and convolution layers are well-suited for detecting these patterns. 
- Parameter Sharing: CNNs use the same kernel across different time steps, which reduces the number of parameters compared to fully connected layers, making the model more efficient and easier to train. 
- Translation Invariance: The model can recognize patterns regardless of their position in the time series, which is useful for detecting repeating patterns such as periodic events. 
''')
st.divider()

st.subheader('Training 1D CNN Model')
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

with st.echo():
    # Build 1D CNN Model
    model_CNN = Sequential([
        Input(shape=(window_size, 1)),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        Conv1D(filters=32, kernel_size=3, activation="relu"),
        Flatten(),
        Dense(50, activation="relu"),
        Dropout(0.3),
        Dense(1)
    ])

    # Compile Model with ADAM optimizer
    model_CNN.compile(optimizer=Adam(), loss="mse")

    # Display Model Summary
    model_CNN.summary()

    # Start tracking total execution time
    cnn_start_time = time.time()

    # Train the Model
    history = model_CNN.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    cnn_model_duration = time.time() - cnn_start_time

if 'cnn_time' not in st.session_state:
    st.session_state['cnn_time'] = cnn_model_duration
else :
    st.session_state['cnn_time'] = cnn_model_duration

with st.container(border=True):
    model_CNN.summary(print_fn=lambda x: st.text(x))
    st.text(f"CNN model training completed in {cnn_model_duration:.2f} seconds")
st.divider()

st.subheader('📊 Training & Validation Loss Over Epochs of 1D CNN')
st.line_chart(pd.DataFrame(history.history), y=['loss', 'val_loss'], color=['#0000ff', '#FF0000'])
st.write('''
         The validation loss is consistently lower than or close to the training loss throughout the epochs,
          and there is no clear upward trend in validation loss after the initial drop. This suggests that the model is not overfitting.
          The fluctuations might indicate some instability, but it doesn't meet the classic overfitting criteria.
          However, the training loss stabilizes at a relatively high value (1000-1500 MSE),
          and the validation loss is also not very low (500-1000 MSE).
          This could indicate the model might not have enough capacity or training epochs to fully capture the data's complexity,
          especially if the task requires lower error.
         In conclusion, the model appears to have a good fit but might be slightly underfitting if the task demands a lower MSE. Increasing model complexity or training duration could help.
''')
st.divider()

st.subheader('📈 Predict Closing Prices using 1D CNN Model')
with st.echo():
    # Predict the values using the trained model
    y_pred = model_CNN.predict(X_test)

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

if 'cnn_rmse' not in st.session_state:
    st.session_state['cnn_rmse'] = rmse
else :
    st.session_state['cnn_rmse'] = rmse

if 'cnn_mae' not in st.session_state:
    st.session_state['cnn_mae'] = mae
else :
    st.session_state['cnn_mae'] = mae


with st.container(border=True):
    st.write(f"RMSE on test set: {rmse:.4f}")
    st.write(f"MAE on test set: {mae:.4f}")
st.divider()

st.subheader('📊 Actual vs. Predicted Closing Prices')
df_predict = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred.flatten()
})
st.line_chart(df_predict, y=['Actual', 'Predicted'], x_label='Time Step', y_label='Closing Price', color=['#0000ff', '#FF0000'])

st.divider()

st.subheader('📈 Short-Term Dependencies')
st.write('''
         - The 1D-CNN appears to perform reasonably well in certain segments. The predicted values closely follow the general trend of the actual values, with the red line tracking the blue line's fluctuations to some extent.
         - However, there are noticeable deviations, particularly in the amplitude and timing of peaks and troughs, indicating that the model struggles to precisely capture rapid changes or short-term fluctuations.
         - This suggests that while the model can approximate short-term dependencies, its resolution might be limited, possibly due to insufficient temporal resolution in the convolutional filters or inadequate hyperparameter tuning.
''')
st.divider()

st.subheader('📈 Evaluation Based on RMSE and MAE Values')
st.write(f'''
         - The RMSE on the test set is {rmse:.4f}, which is relatively high compared to the range of values (approximately 240-340), suggesting that the model's predictions have substantial errors on average.
         - The MAE on the test set is {mae:.4f}, which is slightly lower than the RMSE, indicating that while the average error is moderate, there are some large individual errors contributing to the higher RMSE.
         - These error metrics (RMSE and MAE) are significant relative to the data scale, implying that the model's predictions are not highly accurate.
''')
st.divider()

st.subheader('❔ Can 1D CNN effectively capture the Long-Term Trends?')
st.write('''
         - The predicted values (red line) show a general upward trend that aligns with the actual values (blue line) over the 300 time steps, suggesting that the 1D-CNN can capture some aspects of the long-term trend.
         - However, the large discrepancies in specific time steps and the inability to match the exact amplitude and timing of peaks indicate that the model does not effectively capture the full complexity of long-term trends.
         - The high RMSE and MAE values further support this, as they reflect consistent errors that prevent the model from reliably predicting the long-term behavior across the entire test set.
''')
st.divider()

st.subheader('⁉️ Why It Struggles with Long-Term Trends?')
st.write('''
         - The 1D-CNN architecture, with only two Conv1D layers (filters=64 and 32, kernel_size=3), may have a limited receptive field, making it challenging to model long-term dependencies that require a broader context.
         - The lack of mechanisms like dilated convolutions or additional recurrent layers (e.g., LSTM) to model extended temporal relationships likely contributes to this limitation.
         - The high error metrics suggest that the model may not be adequately trained or regularized to generalize over longer sequences, potentially due to overfitting or insufficient data preprocessing to highlight long-term patterns.
''')

