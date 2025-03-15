import streamlit as st
import pandas as pd
import pathlib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import GRU, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

st.set_page_config(layout='wide')
st.title('🚤 Gated Recurrent Unit (GRU)')

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
st.subheader('❓ Why Use GRU Model for times series forecasting?')
st.write('''
- **Effective Handling of Long-Term Dependencies**'
    - GRUs are designed to mitigate the **vanishing gradient problem** that plagues `SimpleRNN` models. They achieve this through **two** gates: the **update gate** and the **reset gate**.
    - The **update gate** determines how much of the past information to pass along to the future, allowing the model to retain long-term dependencies (e.g., a sustained downward trend in closing prices over 200 time steps, as seen in your plots).
    - The **reset gate** decides how much of the past information to forget, helping the model focus on relevant patterns while discarding noise or irrelevant data.
- **Computational Efficiency Compared to LSTMs**
    - GRUs have a simpler architecture with only two gates (update and reset), reducing the number of parameters and computational overhead compared to LSTMs. For example, the GRU model has two layers with 80 and 40 units (120 total units), while the LSTM had 100 and 50 units (150 total units).
    - This efficiency makes GRUs faster to train and less memory-intensive, which is beneficial for tasks like yours with a relatively small dataset (1000 data points). The reduced complexity can also lead to faster convergence, although we still used 100 epochs to ensure full learning, as with the LSTM.
''')
st.divider()

st.subheader('Training GRU Model')
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
    # Build the GRU model with Input layer
    model_GRU = Sequential([
        Input(shape=(window_size, 1)),
        GRU(80, activation='relu', return_sequences=True),
        GRU(40, activation='relu'),
        Dense(1)  # Output layer for predicting the next value
    ])

    # Compile the model
    model_GRU.compile(optimizer=Adam(), loss='mse')

    # Print model summary
    model_GRU.summary()

    # Start tracking total execution time
    gru_start_time = time.time()

    # Train the model
    history = model_GRU.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    gru_model_duration = time.time() - gru_start_time

if 'gru_time' not in st.session_state:
    st.session_state['gru_time'] = gru_model_duration
else :
    st.session_state['gru_time'] = gru_model_duration


with st.container(border=True):
    model_GRU.summary(print_fn=lambda x: st.text(x))
    st.text(f"GRU model training completed in {gru_model_duration:.2f} seconds")
st.divider()

st.subheader('📊 Training & Validation Loss Over Epochs of GRU Model')
st.line_chart(pd.DataFrame(history.history), y=['loss', 'val_loss'], color=['#0000ff', '#FF0000'])
st.write('''Similar to LSTM and RNN, the model might be underfitting if the validation loss should be lower for the task. The large initial training loss and stable validation loss suggest the model learns quickly but may need adjustment (e.g., more layers or features).''')
st.divider()

st.subheader('📈 Predict Closing Prices using GRU Model')
with st.echo():
    # Predict on the test set
    y_pred = model_GRU.predict(X_test)

    # Inverse transform the predictions and actual values to original scale
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)

if 'gru_rmse' not in st.session_state:
    st.session_state['gru_rmse'] = rmse
else :
    st.session_state['gru_rmse'] = rmse

if 'gru_mae' not in st.session_state:
    st.session_state['gru_mae'] = mae
else :
    st.session_state['gru_mae'] = mae

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
st.divider()

st.subheader('❔ How does the GRU compare to LSTM in terms of computational efficiency and forecasting performance?')
st.write(f'''
- **Computational Efficiency**: refers to the resources (e.g., time, memory) required to train and run the model, which is influenced by the model's architecture, number of parameters, and training dynamics.
    - **Architectural Differences**
        - **GRU**
            - GRUs have a simpler architecture with **two** gates: the **update gate** (determines how much past information to carry forward) and the **reset gate** (decides how much past information to forget).
            - At each time step, a GRU updates its hidden state using these gates, combining the roles of forgetting and updating into a single update gate operation. This reduces the number of computations compared to LSTM.
        - **LSTM**
            - LSTMs are more complex, with **three** gates: the **forget gate** (decides what to forget), the **input gate** (decides what new information to add), and the **output gate** (determines what to output). Additionally, LSTMs maintain a cell state separate from the hidden state to preserve long-term memory.
        - The GRU is more computationally efficient than the LSTM due to its simpler architecture, fewer parameters (34,241 vs. 71,051), and reduced computational overhead per time step. This translates to faster training and inference times and lower memory usage, making GRUs a practical choice when efficiency is a priority.
- **Forecasting Performance**
    - From the GRU plot, the predicted values (red line) generally follow the trend of the actual values (blue line), with RMSE {rmse:4f} and MAE {mae:.4f} accounting for about 6-8% of the data range (220-340), indicating decent forecasting ability but with noticeable deviations, especially in the amplitude of peaks and troughs.
    - LSTM's forecasting results are very close to GRU's, with minimal differences in error metrics. LSTM, with its stronger memory capacity, may have a slight edge in capturing long-term dependencies, particularly in complex sequences, while GRU is better suited for short-term dependencies. However, GRU's more pronounced overfitting (training loss near 0, validation loss at 500-600) suggests slightly weaker generalization compared to LSTM.
    - Overall, LSTM slightly outperforms GRU in forecasting performance, particularly in terms of stability and capturing long-term trends.
''')
st.divider()