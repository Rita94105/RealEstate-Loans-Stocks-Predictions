import streamlit as st

st.set_page_config(
    page_title="Real Estate, Loan Approvals, and Stock Market Trends.",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏠 Predictions for Real Estate, Loan Approvals, and Stock Market Trends.")

st.markdown(
    """
    This project seeks to deepen our understanding of real estate pricing, loan approvals, and stock market trends by analyzing three distinct datasets and building predictive models using **machine learning**, **statistical analysis**, and **data visualization**. 
    By leveraging these techniques, we aim to uncover meaningful insights, deliver accurate predictions, and showcase the power of **data-driven decision-making** across these domains.

    ### ⭐ **Dataset1.csv: data related to the housing market.** 
    - The goal is to predict house prices based on various factors such as the square footage of the house, number of bedrooms and bathrooms, the age of the house, and proximity to important amenities like the city center, schools, and grocery stores. 
    - Other attributes such as walkability score, crime rate, and property tax rate are also considered, along with the neighborhood's median income. 
    - The target variable in this dataset is the house price, and the aim is to develop a model that can accurately predict house prices based on the provided features. 
    - In this project, we tried to build **two multiple linear regression models** on the correlation-based subsets to assess feature impact on prediction accuracy and a **Lasso Regression model** to the full dataset to compare and find the best model.

    ### ⭐ **Dataset2.csv: centered around loan status prediction.**
    - This dataset contains information related to loan applicants, including their credit scores, income, requested loan amount, loan term, debt-to-income ratio, and employment details. 
    - The target variable here is the loan status, which classifies the loan application into three categories: fully approved, conditionally approved, or rejected. 
    - In the project, we tried a suite of machine learning techniques such as **Logistic Regression**, **K-Nearest Neighbors (KNN)**, **Random Forest (RF)**, **Single-Layer Neural Network (Perceptron)** and **Multi-Layer Perceptron (MLP)**
    - This multi-algorithm strategy aims to optimize classification accuracy and uncover the most effective predictors of loan approval outcomes.
    
    ### ⭐ **StockData.csv: a time series dataset designed to simulate stock market behavior over a period of 1000 days.**
    - starting from January 1, 2023.
    - This dataset includes critical market features such as opening price, daily high and low prices, closing price, and trading volume.
    - The objective is to analyze trends, identify patterns, and develop a model that can predict stock price movements.
    - To address this, we deploy a suite of advanced **time series modeling techniques**:
        1. **1D-CNN** for extracting local temporal features.
        2. **RNN** to model sequential dependencies.
        3. **LSTM** and **GRU** for capturing long-term patterns and managing memory efficiently.
        4. **Transformer** to leverage attention mechanisms for dynamic trend detection.
    - By training and comparing these models, we seek to determine the most effective approach for **stock price prediction** and gain insights into **market behavior drivers**.
"""
)