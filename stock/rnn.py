import streamlit as st
import pandas as pd
import pathlib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

st.set_page_config(layout='wide')
st.title('🛩️ Recurrent Neural Networks (RNN)')

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
st.subheader('❓ Why Use RNN for times series forecasting?')
st.markdown('''Time series data, such as stock closing prices, temperature, or traffic flow, has a key characteristic: temporal dependencies. This means that the current value is often influenced by past values, and these dependencies may span multiple time steps. Traditional neural networks (like fully connected networks) struggle to handle such temporal dependencies because they assume inputs are independent and identically distributed (i.i.d.) and do not account for sequence or temporal context. 

RNNs (Recurrent Neural Networks) are specifically designed to address this issue, and here's why they are suitable for time series forecasting:
- **Memory Capability**
    - RNNs have a recurrent structure that allows them to pass information from one time step to the next, effectively creating a form of "memory." This enables the model to capture both short-term and long-term dependencies in the time series.
- **Flexibility with Sequences**
    - RNNs can handle variable-length sequences, which is ideal for time series forecasting since different tasks may involve different lengths of historical data.''')
st.divider()

st.subheader('Training RNN Model')
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
    # Build the RNN model
    model_RNN = Sequential([
        Input(shape=(window_size, 1)),
        SimpleRNN(40, activation='relu'),
        Dense(1)  # Output layer for predicting the next value
    ])

    # Compile the model
    model_RNN.compile(optimizer=Adam(), loss='mse')

    # Display Model Summary
    model_RNN.summary()

    # Start tracking total execution time
    rnn_start_time = time.time()

    # Train the model
    history = model_RNN.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    rnn_model_duration = time.time() - rnn_start_time

if 'rnn_time' not in st.session_state:
    st.session_state['rnn_time'] = rnn_model_duration
else:
    st.session_state['rnn_time'] = rnn_model_duration


with st.container(border=True):
    model_RNN.summary(print_fn=lambda x: st.text(x))
    st.text(f"RNN model training completed in {rnn_model_duration:.2f} seconds")
    
st.divider()
st.subheader('📊 Training & Validation Loss Over Epochs of RNN')
st.line_chart(pd.DataFrame(history.history),y=["loss","val_loss"],color=["#0000ff","#FF0000"])
st.write('''
The training loss drops to near 0, which might suggest the model is fitting the training data perfectly,
          but the validation loss stabilizes without a significant increase.
          This could indicate underfitting or a very good fit rather than overfitting,
          as the validation loss does not diverge from the training loss.
          However, the very low training loss might warrant further investigation (e.g., checking for data leakage or model capacity).
         In conclusion, the model might be underfitting due to the gap between training and validation loss,
          or it could be overfitting if the near-0 training loss is misleading (e.g., due to data issues).
          Further investigation into the dataset and model architecture is needed.
''')

st.divider()
st.subheader('📈 Predict Closing Prices using RNN Model')
with st.echo():
    # Predict on the test set
    y_pred = model_RNN.predict(X_test)

    # Inverse transform the predictions and actual values to original scale
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)

if 'rnn_rmse' not in st.session_state:
    st.session_state['rnn_rmse'] = rmse
else:
    st.session_state['rnn_rmse'] = rmse

if 'rnn_mae' not in st.session_state:
    st.session_state['rnn_mae'] = mae
else:
    st.session_state['rnn_mae'] = mae

with st.container(border=True):
    st.write(f"RMSE on test set: {rmse:.4f}")
    st.write(f"MAE on test set: {mae:.4f}")
st.divider()
st.subheader('📈 Predictions vs Actual Closing Prices')
df_predict = pd.DataFrame({
    "Actual": y_test_original,
    "Predicted": y_pred_original
})
st.line_chart(df_predict, y=['Actual', 'Predicted'] ,x_label='Time Step', y_label='Closing Price', color=['#0000ff', '#FF0000'])
st.divider()

st.subheader('❔ Why Did We Use a Smaller Batch Size Compared to the CNN Model?')
st.write('''
- RNNs process data sequentially, where each time step depends on the previous one due to the recurrent structure. This sequential dependency makes training more sensitive to the order and flow of data, and a smaller batch size (e.g., 16) allows for more frequent weight updates. This helps the model adjust its parameters more responsively to the temporal patterns in the time series.
- In contrast, CNNs (including 1D-CNNs) operate on fixed-size windows or patches of data and are less dependent on the sequential order within a batch. Their convolutional operations are parallelizable, so larger batch sizes (e.g., 32 or 64) are often used to leverage GPU parallelism and stabilize gradient updates.
- In summary, the smaller batch size of 16 for the RNN reflects its sequential processing needs, memory constraints, and the desire for frequent updates to capture temporal dependencies, contrasting with the CNN’s ability to use larger batches due to its parallelizable architecture.
''')
st.divider()

st.subheader('⁉️ Why Might RNN Struggle with Long-Term Dependencies?')
st.write('''
The SimpleRNN’s limitations due to the vanishing gradient problem, lack of gating mechanisms, and a small 10-step window hinder its ability to capture long-term dependencies, as evidenced by lags and amplitude mismatches in the plot.
- **Vanishing Gradient Problem**
    - During backpropagation through time (BPTT), the gradients of the loss with respect to the weights are propagated backward across multiple time steps. In a SimpleRNN, the recurrent weight matrix is repeatedly multiplied, and if the eigenvalues of this matrix are less than 1, the gradients diminish exponentially with the number of time steps. This makes it difficult for the model to learn dependencies that are far apart in the sequence.
    - For example, if the closing price trend depends on data from 50 time steps ago, the SimpleRNN may fail to propagate this information effectively due to gradient vanishing.
- **Limited Memory**
    - The hidden state in a SimpleRNN is a simple linear combination of the previous hidden state and the current input, with no mechanism to selectively forget or prioritize information. This limits its ability to retain relevant information over long periods, especially in the presence of noise or irrelevant data points.
    - A window size of 10 further restricts the model's context, meaning it can only “remember” up to 10 steps unless additional mechanisms (e.g., stacking layers) are used.
- **Activation Function Impact**
    - The use of ReLU activation in the SimpleRNN can exacerbate the vanishing gradient issue. ReLU sets negative values to 0, which can interrupt the flow of gradients during BPTT, especially for long sequences. Traditional activations like tanh or sigmoid are more stable but still suffer from vanishing gradients in deep or long sequences.
- **Architectural Simplicity**
    - SimpleRNN lacks the gating mechanisms found in advanced variants like LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit). These mechanisms (e.g., forget gates, input gates) allow LSTMs and GRUs to selectively retain or discard information over long time spans, making them better suited for long-term dependencies.
''')
st.divider()

st.subheader('❓ How Is This Reflected in the Evaluation Metrics?')
st.write(f'''
    1. **High RMSE and MAE**
         - The RMSE {rmse:.4f} and MAE {mae:.4f} values are relatively high compared to the data range (approximately 220-340), indicating significant prediction errors.
    2. **Plot Analysis (Actual vs. Predicted Values)**
         - The plot shows noticeable deviations between the predicted values (red line) and actual values (blue line), particularly in the timing and amplitude of peaks and troughs.

    These elevated error metrics reflect the RNN's difficulty in capturing long-term dependencies, as it fails to accurately predict trends over distant time steps, resulting in accumulated discrepancies between predicted and actual values across the time series.
''')