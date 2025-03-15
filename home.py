import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Real Estate, Loan Approvals, and Stock Market Trends.",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏠 Predictions for Real Estate, Loan Approvals, and Stock Market Trends.")

st.markdown(
    '''
    This project seeks to deepen our understanding of real estate pricing, loan approvals, and stock market trends by analyzing three distinct datasets and building predictive models using **machine learning**, **statistical analysis**, and **data visualization**. 
    By leveraging these techniques, we aim to uncover meaningful insights, deliver accurate predictions, and showcase the power of **data-driven decision-making** across these domains.

    ### ⭐ **Dataset1.csv: data related to the housing market.** 
    The goal is to predict house prices based on various factors. Key features included:
    - Square footage of the house
    - Number of bedrooms and bathrooms
    - Age of the house
    - Proximity to important amenities like the city center, schools, and grocery stores
    - Additional attributes such as walkability score, crime rate, property tax rate, and the neighborhood's median income
    
    A common challenge in this analysis was dealing with multicollinearity among features, such as square footage and number of bedrooms, which could skew the regression results. 
    To address this, the project used correlation-based feature selection to create subsets and applied Lasso Regression for regularization, effectively reducing the impact of less significant variables and improving model interpretability.
''')
d1_df= pd.DataFrame({
    'Model Tpye': ['Multiple Linear Regression', 'Lasso Regression'],
    'Approach': ['Correlation-based subsets', 'Full dataset'],
    'Challenges': ['Multicollinearity', 'Feature selection, Overfitting']
})
st.dataframe(d1_df)
st.divider()

st.markdown('''
    ### ⭐ **Dataset2.csv: centered around loan status prediction.**
    with the target variable classifying loan applications into three categories: fully approved, conditionally approved, or rejected. The dataset included:
    - Credit scores
    - Income
    - Requested loan amount
    - Loan term
    - Debt-to-income ratio
    - Employment details
    
    To tackle this multi-class classification problem, the project employed a suite of machine learning techniques, including:
''')
d2_df = pd.DataFrame({
    'Model Type': ['Logistic Regression', 'K-Nearest Neighbors (KNN)', 'Random Forest', 'Perceptron, MLP'],
    'Use Case': ['Baseline classification', 'Instance-based learning', 'Ensemble method', 'Neural networks for complexity'],
    'Challenges': ['Interpretability', 'Computational efficiency', 'Overfitting', 'Non-linear relationships']
})
st.dataframe(d2_df)
st.divider()
st.markdown('''
    ### ⭐ **StockData.csv: a time series dataset designed to simulate stock market behavior over a period of 1000 days.**
    - starting from January 1, 2023.
    - This dataset includes critical market features such as opening price, daily high and low prices, closing price, and trading volume.
    - The objective is to analyze trends, identify patterns, and develop a model that can predict stock price movements.
    To address this, the project deploy a suite of advanced **time series modeling techniques**:
''')
d3_df = pd.DataFrame({
    'Model Type': ['1D CNN', 'RNN', 'LSTM, GRU', 'Transformer'],
    'Purpose': ['Local feature extraction', 'Sequential dependencies', 'Long-term memory', 'Dynamic trend detection'],
    'Challenge Addressed': ['Short-term patterns', 'Memory in sequences', 'Trends and seasonality', 'Attention to relevant data']
})
st.dataframe(d3_df)