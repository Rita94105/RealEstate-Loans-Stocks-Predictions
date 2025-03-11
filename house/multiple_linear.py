import streamlit as st
import numpy as np
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
# Load cleaned data
df_cleaned = pd.read_csv(CURRENT_DIR / 'Dataset1_Cleaned.csv')

st.subheader('✂️ Split the data into train(70%) and test(30%)')

st.markdown('''
            - Build three different subsets of our data using two different sets of features based on correlation thresholds in previous question, as well as the original dataset with all features. 
            - For each subset, split the data into train and test and hold out 30% of observations as the test set. 
            - Pass random_state=42 to train_test_split to ensure you get the same train and tests sets as the solution and normalize (z-normalization) the data splits.
            ''')

with st.echo():

    # Create three datasets
    correlations = df_cleaned.corr()["House_Price"]
    features_002 = correlations[correlations > 0.02].index.tolist()
    features_004 = correlations[correlations > 0.04].index.tolist()

    df_all = df_cleaned.copy()
    df_002 = df_cleaned[features_002]
    df_004 = df_cleaned[features_004]

    # Function to split and normalize data
    def split_and_normalize(df):
        X = df.drop(columns=["House_Price"])
        y = df["House_Price"]
    
        # Train-test split (30% test set)
        X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=42
        )

        # Apply Z-score normalization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    # Apply function to all datasets
    X_train_all, X_test_all, y_train_all, y_test_all = split_and_normalize(df_all)
    X_train_002, X_test_002, y_train_002, y_test_002 = split_and_normalize(df_002)
    X_train_004, X_test_004, y_train_004, y_test_004 = split_and_normalize(df_004)

with st.container(border=True):
    st.write(f"Dataset (All features): Train {X_train_all.shape}, Test {X_test_all.shape}")
    st.write(f"Dataset (corr > 0.02): Train {X_train_002.shape}, Test {X_test_002.shape}")
    st.write(f"Dataset (corr > 0.04): Train {X_train_004.shape}, Test {X_test_004.shape}")

st.subheader('🔧 Build two different multiple linear regression models')
st.markdown('''
        - Using the subsets made by two different thresholds, 0.02 and 0.04, and train them on their normalized training sets.
        - Build and fit a Lasso Regression model to the training data using all features in the dataset. The penalization parameter is set to 0.5.
''')

with st.echo():# Initialize the linear regression model
    lr_model_002 = LinearRegression().fit(X_train_002, y_train_002)
    lr_model_004 = LinearRegression().fit(X_train_004, y_train_004)

    # Initialize the Lasso Regression model with alpha=0.5
    lasso_model = Lasso(alpha=0.5)

    # Fit the model to the training data (using all features)
    lasso_model.fit(X_train_all, y_train_all)

st.subheader('💯 Evaluate the models on the test set and compare the models using R² and RMSE')

with st.echo():
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
    
        # Calculate R²
        r2 = r2_score(y_test, y_pred)
    
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
        return r2, rmse

    r2_002, rmse_002 = evaluate_model(lr_model_002, X_test_002, y_test_002)
    r2_004, rmse_004 = evaluate_model(lr_model_004, X_test_004, y_test_004)
    r2_lasso, rmse_lasso = evaluate_model(lasso_model, X_test_all, y_test_all)

with st.container(border=True):
    st.write(f"Multiple Linear Regression (corr > 0.02): R² = {r2_002:.4f}, RMSE = {rmse_002:.4f}")
    st.write(f"Multiple Linear Regression (corr > 0.04): R² = {r2_004:.4f}, RMSE = {rmse_004:.4f}")
    st.write(f"Lasso Regression (all features): R² = {r2_lasso:.4f}, RMSE = {rmse_lasso:.4f}")

st.header('🚀 Conclusion')
st.container(border=True).write('''
- Since the Lasso Regression model has the highest R² and the lowest RMSE, it performs the best among the three models.'
- The next steps for further improvement could include:
    - Fine-tuning the Lasso Model: Optimize the regularization parameter (alpha)
    - Remove less relevant or redundant features: Lasso helps with feature selection, but additional techniques like Recursive Feature Elimination (RFE) or SHAP analysis could further refine feature importance.
    - Explore Alternative Models
    - Use Cross-Validation                            
''')