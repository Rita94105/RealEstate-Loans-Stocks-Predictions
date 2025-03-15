import streamlit as st
import pandas as pd
import pathlib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Dense, Add
import time

st.set_page_config(layout='wide')
st.title('🛸 Transformers')

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
st.subheader('❓ Why Use Transformer for times series forecasting?')
st.write('''Unlike recurrent neural networks (RNNs) such as LSTMs and GRUs, which process sequences sequentially and can struggle with long-term dependencies due to vanishing gradients, Transformers use attention mechanisms to capture relationships between all time steps simultaneously. This makes them particularly well-suited for time series data like closing prices, which may exhibit dependencies across various time scales (e.g., short-term fluctuations, weekly seasonality, or long-term trends).''')
st.divider()

st.subheader('Training Transformer Model')
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
    # Build the Transformer model
    def build_transformer_model(window_size):
        inputs = Input(shape=(window_size, 1))
    
        # MultiHeadAttention layer with residual connection
        attention_output = MultiHeadAttention(key_dim=32, num_heads=2)(inputs, inputs)  # Self-attention
        attention_output = Add()([inputs, attention_output])  # Residual connection
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
        # Global average pooling
        pooled_output = GlobalAveragePooling1D()(attention_output)
    
        # Fully connected layer for prediction
        outputs = Dense(1)(pooled_output)
    
        # Define the model
        model = Model(inputs=inputs, outputs=outputs)
        return model

    # Create and compile the model
    model_transformer = build_transformer_model(window_size)
    model_transformer.compile(optimizer=Adam(), loss='mse')

    # Print model summary
    model_transformer.summary()

    # Start tracking total execution time
    transformer_start_time = time.time()

    # Train the model
    history = model_transformer.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    transformer_model_duration = time.time() - transformer_start_time

if 'transformer_time' not in st.session_state:
    st.session_state['transformer_time'] = transformer_model_duration
else :
    st.session_state['transformer_time'] = transformer_model_duration

with st.container(border=True):
    model_transformer.summary(print_fn=lambda x: st.text(x))
    st.write(f"Transformer training completed in: {transformer_model_duration:.2f} seconds")
st.divider()

st.subheader('📊 Training & Validation Loss Over Epochs of Transformer')
st.line_chart(pd.DataFrame(history.history), y=['loss', 'val_loss'], color=['#0000ff', '#FF0000'])
st.write('''The Transformer model’s training loss decreases slowly to 3000, but the validation loss stabilizes at 7000, indicating significant overfitting and large prediction errors, with poor ability to capture data patterns. It is recommended to adjust the model architecture, add regularization, or extend training time to improve performance.''')
st.divider()

st.subheader('📈 Predict Closing Prices using 1D CNN Model')
with st.echo():
    # Predict on the test set
    y_pred = model_transformer.predict(X_test)

    # Inverse transform the predictions and actual values to original scale
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)

if 'transformer_rmse' not in st.session_state:
    st.session_state['transformer_rmse'] = rmse
else :
    st.session_state['transformer_rmse'] = rmse

if 'transformer_mae' not in st.session_state:
    st.session_state['transformer_mae'] = mae
else :
    st.session_state['transformer_mae'] = mae

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
st.write("The Transformer model exhibits extremely poor forecasting performance, with predicted values remaining nearly constant at 0 and showing no correlation with actual values, accompanied by very high RMSE and MAE, indicating a failure to capture time series patterns. It is recommended to review data quality, adjust hyperparameters, or consider a more suitable model architecture.")
st.divider()

st.subheader('🔍 Other hyperparameters relevant to transformers')
st.write('''
According to Tensorflow Transformer Model, there are still some hyperparameters that we can be tuned:
- **Model related hyperparameters**
    - `num_layers`: Number of layers in the encoder and decoder
    - `d_model`: Model dimension for both encoder and decoder
    - `num_heads`: Number of attention heads for self-attention mechanisms
    - `dff`: Number of units in the hidden layer of the feed-forward networks
    - `input_vocab_size`: Size of the input vocabulary (encoder)
    - `target_vocab_size`: Size of the output vocabulary (decoder)
    - `dropout_rate`: Dropout rate for regularization
- **Training related hyperparameters**
    - `learning_rate`: Defaults to 0.001
    - `batch_size`: Number of samples per batch of computation. Default to 32.
    - `epochs`: Number of epochs to train the model
    - `optimizer`        
''')
