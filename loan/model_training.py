import streamlit as st
import pathlib
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

st.title('✈️ Model Training')

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

# Load data
df = pd.read_csv(CURRENT_DIR / 'Dataset2.csv')
categorical_features = df.select_dtypes(include=["object"]).columns
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

st.subheader('🔧 fit the following models to the training samples')
st.markdown('''
    - Logistic Regression
    - K-Nearest Neighbors (KNN) with K equals to 3
    - Random Forest (RF) that consists of 5 base decision trees with the maximum depth of 5
    - Single-Layer Neural Network (Perceptron) with stochastic gradient descent (SGD) optimizer and a learning rate of 0.1, run the model for 10 iterations/epochs.
''')
X2 = df_encoded.drop(columns=['Loan_Status'])
y2 = df_encoded['Loan_Status']
# Split the data into train, validation and test and hold out 20% and 10% of observations as the validation and test set
X2_train, X2_temp, y2_train, y2_temp = train_test_split(X2, y2, test_size=0.3, random_state=42)
X2_val, X2_test, y2_val, y2_test = train_test_split(X2_temp, y2_temp, test_size=1/3, random_state=42)

# Normalize Data (Z-score normalization)
scaler = StandardScaler()
X2_train = scaler.fit_transform(X2_train)
X2_val = scaler.transform(X2_val)
X2_test = scaler.transform(X2_test)

with st.echo():
    # Function to measure training time
    def time_model(model, X_train, y_train):
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        return (end_time - start_time) * 1000 

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_time = time_model(log_reg, X2_train, y2_train)

    # KNN (K=3)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn_time = time_model(knn, X2_train, y2_train)

    # Random Forest (5 trees, max depth = 5)
    rf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=42)
    rf_time = time_model(rf, X2_train, y2_train)

    # Single-Layer Neural Network (Perceptron) with stochastic gradient descent (SGD) optimizer (learning rate = 0.1, 10 iterations/epochs)
    mlp = MLPClassifier(hidden_layer_sizes=(), solver='sgd', learning_rate_init=0.1, max_iter=10, random_state=42)
    mlp_time = time_model(mlp, X2_train, y2_train)

st.subheader('📊 Training time(ms)')
col1, col2, col3, col4 = st.columns(4,border=True)
with col1:
    st.metric("Logistic Regression", f"{log_reg_time:.2f}")
with col2:
    st.metric("K-Nearest Neighbors", f"{knn_time:.2f}")
with col3:
    st.metric("Random Forest", f"{rf_time:.2f}")
with col4:
    st.metric("Single-Layer Neural Network", f"{mlp_time:.2f}")

st.subheader('📈 Feature importance scores by Random Forest model')

with st.echo():
    # Create a DataFrame with feature names and their corresponding importance scores
    feature_importances = rf.feature_importances_
    features = X2.columns.tolist()

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    })

    # Sort features by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot a horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance - Random Forest')
    plt.gca().invert_yaxis()  # To show the most important feature at the top
st.pyplot(plt.gcf())

st.subheader('📍 Select the important features from most to least important until the accumulated relative importance score reaches 90% ')

with st.echo():
    cumulative_importance = np.cumsum(importance_df['Importance'])
    selected_features = importance_df[cumulative_importance <= 0.9]

st.table(selected_features)

st.subheader('🔧 Build a Multi-Layer Perceptron (MLP)')
st.markdown('''
    - Two hidden layers (H1, H2), with 50 and 100 neurons/units in H1 and H2, respectively.
    - Use tanh function as the activation function for hidden layers.
    - Use a proper acitivation function for the output layer.
    - Use Stochastic gradient descent optimizer with a learning rate of 0.1.
    - Run the model for 10 iterations/epochs
''')

with st.echo():
    # Multi-Layer Perceptron (H1: 50, H2:100, activation function: tanh, learning rate:0.1, max_iterations:10)
    mlp = MLPClassifier(hidden_layer_sizes=(50, 100), activation='tanh', solver='sgd', learning_rate_init=0.1, max_iter=10, random_state=42)

    training_loss = []
    validation_loss = []

    mlp_train_start_time = time.time()
    for i in range(mlp.max_iter):
        mlp.partial_fit(X2_train, y2_train, classes=np.unique(y2_train)) 
        train_loss = mlp.loss_
        val_loss = log_loss(y2_val, mlp.predict_proba(X2_val))
    
        training_loss.append(train_loss)
        validation_loss.append(val_loss)

    mlp_training_time = (time.time() - mlp_train_start_time) * 1000 

    # Plot learning curves
    epochs = range(1, mlp.max_iter + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_loss, marker='o', label='Training Loss', color='blue')
    plt.plot(epochs, validation_loss, marker='o', linestyle='dashed', label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve of MLP')
    plt.legend()
    plt.grid(True)
st.pyplot(plt.gcf())