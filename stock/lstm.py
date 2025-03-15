import streamlit as st
import pandas as pd
import pathlib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import io

st.set_page_config(layout='wide')
st.title('🚁 Long Short-Term Memory (LSTM)')

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
st.subheader('❓ Why Use LSTM Model for times series forecasting?')
st.write('''
- **Handling Long-Term Dependencies**
    - **Challenge in Time Series**: Time series data often has dependencies that span multiple time steps. For example, a stock's closing price might be influenced by a trend that started 30 days ago or a recurring weekly pattern. Capturing these long-term dependencies is critical for accurate forecasting.
    - Unlike the `SimpleRNN`, which struggles with long-term dependencies due to the vanishing gradient problem (as discussed earlier), LSTMs are explicitly designed to address this issue.
    - LSTMs use a memory cell and three gates (forget gate, input gate, and output gate) to selectively retain or discard information over long time periods. The forget gate decides which information to discard from the memory, the input gate decides what new information to add, and the output gate determines what to output at each time step. This allows LSTMs to maintain and update a “memory” of past events, even across many time steps.
- **Selective Memory and Noise Handling**
    - **Challenge in Time Series**: Time series data often contains noise or irrelevant fluctuations (e.g., daily volatility in stock prices that doesn't affect the overall trend). A model needs to distinguish between noise and meaningful patterns.
    - The forget gate in an LSTM allows the model to selectively forget irrelevant information (e.g., short-term noise) while retaining important long-term patterns (e.g., a sustained upward trend). This selective memory makes LSTMs more robust to noise compared to SimpleRNN, which treats all past information equally and can get overwhelmed by irrelevant data.
''')
st.divider()

st.subheader('Training LSTM Model')
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
    # Build the LSTM model
    model_LSTM = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(100, activation='relu', return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model_LSTM.compile(optimizer=Adam(), loss='mse')

    # Print model summary
    model_LSTM.summary()

    # Start tracking total execution time
    lstm_start_time = time.time()

    # Train the model
    history = model_LSTM.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )
    lstm_model_duration = time.time() - lstm_start_time

if 'lstm_time' not in st.session_state:
    st.session_state['lstm_time'] = lstm_model_duration
else :
    st.session_state['lstm_time'] = lstm_model_duration

with st.container(border=True):
    model_LSTM.summary(print_fn=lambda x: st.text(x))
    st.text(f"LSTM model training completed in {lstm_model_duration:.2f} seconds")
st.divider()

st.subheader('📊 Training & Validation Loss Over Epochs of LSTM')
st.line_chart(pd.DataFrame(history.history), y=['loss', 'val_loss'], color=['#0000ff', '#FF0000'])
st.write('''Similar to the RNN, the training loss reaches near 0, while the validation loss stabilizes at a higher value. There is no significant increase in validation loss, suggesting the model is not overfitting. The gap between training and validation loss might indicate the model has learned the training data well but generalizes reasonably, though the low training loss again raises the possibility of underfitting or an overly simple model.''')
st.divider()

st.subheader('📈 Predict Closing Prices using LSTM Model')
with st.echo():
    # Predict on the test set
    y_pred = model_LSTM.predict(X_test)

    # Inverse transform the predictions and actual values to original scale
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)

if 'lstm_rmse' not in st.session_state:
    st.session_state['lstm_rmse'] = rmse
else :
    st.session_state['lstm_rmse'] = rmse

if 'lstm_mae' not in st.session_state:
    st.session_state['lstm_mae'] = mae
else :
    st.session_state['lstm_mae'] = mae

with st.container(border=True):
    st.write(f"RMSE on test set: {rmse:.4f}")
    st.write(f"MAE on test set: {mae:.4f}")
st.divider()

st.subheader('📊 Actual vs. Predicted Closing Prices')
df_predict = pd.DataFrame({
    "Actual": y_test_original,
    "Predicted": y_pred_original
})
st.line_chart(df_predict, y=['Actual', 'Predicted'], x_label='Time Step', y_label='Closing Price', color=['#0000ff', '#FF0000'])
st.write('''Similar to the RNN, the training loss reaches near 0, while the validation loss stabilizes at a higher value. There is no significant increase in validation loss, suggesting the model is not overfitting. The gap between training and validation loss might indicate the model has learned the training data well but generalizes reasonably, though the low training loss again raises the possibility of underfitting or an overly simple model.''')
st.divider()

st.subheader('📈 Explanation of the LSTM Model Results')
st.write(f'''
- **Evaluation Metrics Analysis**
    - The RMSE {rmse:.4f} and MAE {mae:.4f} values, relative to the data range (approximately 220-340), account for about 6-8% of the range, indicating that the model has some effectiveness in capturing time series patterns but with limited prediction accuracy and noticeable deviations.
- **Plot Analysis (Actual vs. Predicted Values)**
    - The plot shows the actual values (blue line) and predicted values (red line) over time steps, with the predicted values generally following the overall trend of the actual values, particularly aligning with peaks and troughs to some extent.
The predicted values do not fully match the amplitude and timing of the actual values at certain time steps, suggesting that LSTM still needs improvement in capturing short-term fluctuations and long-term trends, especially in terms of fine-grained accuracy.
''')
st.divider()

st.subheader('⁉️ Why do we use a larger epoch compared to RNN?')
st.write('''
- **Increased Model Complexity**
    - LSTMs are more complex due to their gating mechanisms (forget gate, input gate, and output gate), which introduce additional parameters and computational overhead compared to the simpler recurrent structure of a SimpleRNN.
- **Learning Dynamics of LSTMs**
    - Slower Convergence
        - LSTMs, with their memory cells and gates, have a more sophisticated learning process compared to SimpleRNN. The gates need to learn when to forget, update, or output information, which requires more iterations to stabilize. This slower convergence necessitates a larger number of epochs to ensure the model fully exploits its capacity.
        - In contrast, the SimpleRNN has a simpler update rule (a single hidden state), allowing it to converge faster with fewer epochs.
    - Gradient Flow
        - LSTMs mitigate the vanishing gradient problem, enabling gradients to flow more effectively through time. However, this also means the model can continue learning and refining its weights over more epochs without the gradients diminishing too quickly, unlike the SimpleRNN, where early convergence might occur due to gradient issues.
''')


