import streamlit as st
import pandas as pd
import pathlib
import io

st.title('💡 Comparison of Models ')


st.write('''Based on the provided results from **StockData.csv** for predicting stock price movements, we evaluated five models—**1D CNN, RNN, LSTM, GRU, and Transformer**—across metrics including **Training Time, RMSE** (Root Mean Squared Error), **MAE** (Mean Absolute Error), and Potential, with additional notes on **potential** issues like underfitting.''')

if 'cnn_time' not in st.session_state or 'rnn_time' not in st.session_state or 'lstm_time' not in st.session_state or 'gru_time' not in st.session_state or 'transformer_time' not in st.session_state:
    st.error('If there is no data in the table to show the result, please browser the models page first.', icon="🚨")
else: 
    df = pd.DataFrame(
        {
            'Model': ['1D CNN', 'RNN',  'LSTM', 'GRU', 'Transformer'],
            'Training Time': [st.session_state.cnn_time, st.session_state.rnn_time, st.session_state.lstm_time, st.session_state.gru_time, st.session_state.transformer_time],
            'RMSE': [st.session_state.cnn_rmse, st.session_state.rnn_rmse, st.session_state.lstm_rmse, st.session_state.gru_rmse, st.session_state.transformer_rmse],
            'MAE': [st.session_state.cnn_mae, st.session_state.rnn_mae, st.session_state.lstm_mae, st.session_state.gru_mae, st.session_state.transformer_mae],
            'Potential': ['Underfitting', 'Underfitting', 'Underfitting', 'Underfitting', 'Underfitting']
        }
    )
    config = {"Training Time": "{:.2f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"}
    st.dataframe(df)

st.write('''The **1D CNN** emerges as the best-performing model due to its low **RMSE** and **MAE**, combined with minimal training time, despite the underfitting concern. 
         This suggests that simpler architectures may be more suitable for this 1000-day stock dataset. 
         However, the consistent underfitting across all models indicates potential issues with model complexity, data preprocessing, or hyperparameter tuning. 
         Future improvements could involve adjusting model architectures, increasing data diversity, or refining training strategies to better capture stock market trends.''')
