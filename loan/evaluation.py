import streamlit as st
import pathlib
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import io

st.title('💡 Model Evaluation and Analysis')

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

# Load data
df = pd.read_csv(CURRENT_DIR / 'Dataset2.csv')
# preprocess the data
categorical_features = df.select_dtypes(include=["object"]).columns
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

X2 = df_encoded.drop(columns=['Loan_Status'])
y2 = df_encoded['Loan_Status']
# Split the data into train, validation and test and hold out 20% and 10% of observations as the validation and test set
X2_train, X2_temp, y2_train, y2_temp = train_test_split(X2, y2, test_size=0.3, random_state=42)
X2_val, X2_test, y2_val, y2_test = train_test_split(X2_temp, y2_temp, test_size=1/3, random_state=42)

scaler = StandardScaler()
X2_train = scaler.fit_transform(X2_train)
X2_val = scaler.transform(X2_val)
X2_test = scaler.transform(X2_test)

st.subheader('📝 Confusion matrix, F1-score, Recall, Precision and Accuracy')
with st.echo():
    # Binarize the labels for multi-class ROC (one-vs-rest)
    y2_test_bin = label_binarize(y2_test, classes=[0, 1, 2])

    # Train models
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    rf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=42)
    slp = MLPClassifier(hidden_layer_sizes=(), solver='sgd', learning_rate_init=0.1, max_iter=10, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(50, 100), activation='tanh', solver='sgd', learning_rate_init=0.1, max_iter=10, random_state=42)
    models = [log_reg, knn, rf, slp, mlp]
    model_names = ['Logistic Regression', 'KNN', 'Random Forest', 'Single-Layer Neural Network', 'Multi-Layer Neural Network']

    # Prepare the results dictionary
    results = {}

    # Dictionary to store the test times
    test_times = {}

    # Evaluate models on test set
    for model, name in zip(models, model_names):
        # Fit the model
        model.fit(X2_train, y2_train)

        # Start the timer
        start_time = time.time()
    
        # Predict probabilities (instead of predicted labels)
        y2_pred_prob = model.predict_proba(X2_test)

        # Stop the timer
        end_time = time.time()
    
        # Calculate the elapsed time in milliseconds
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
        # Store the result
        test_times[name] = elapsed_time
    
        # Predict on the test set
        y2_pred = model.predict(X2_test)
    
        # Calculate evaluation metrics
        cm = confusion_matrix(y2_test, y2_pred)
        f1 = f1_score(y2_test, y2_pred, average='weighted', zero_division=1)
        recall = recall_score(y2_test, y2_pred, average='weighted', zero_division=1)
        precision = precision_score(y2_test, y2_pred, average='weighted', zero_division=1)
        accuracy = accuracy_score(y2_test, y2_pred)
        total_auc = 0

        # Compute ROC curve and AUC for each class (one-vs-rest)
        for i, class_label in enumerate([0, 1, 2]):
            fpr, tpr, _ = roc_curve(y2_test_bin[:, i], y2_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} Class {class_label} (AUC = {roc_auc:.2f})')
            total_auc += roc_auc

        # Store results
        results[name] = {
            'Confusion Matrix': cm,
            'F1 Score': f1,
            'Recall': recall,
            'Precision': precision,
            'Accuracy': accuracy,
            'Average AUC': float(total_auc) / 3
        }

with st.container(border=True):
    df = pd.DataFrame()
    for model_name, result in results.items():
        # Display confusion matrix as a plot
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(result['Confusion Matrix'])
        disp.plot(ax=ax)
        ax.set_title(f'Confusion Matrix for {model_name}')
        st.pyplot(fig)

        row = pd.DataFrame({'model':model_name,
                            'F1 Score':result['F1 Score'],
                            'Recall':result['Recall'],
                            'Precision':result['Precision'],
                            'Accuracy':result['Accuracy'],
                            'Test Time (ms)':test_times[model_name],
                            'Average AUC':result['Average AUC']
                            }, index=[0])
        df = pd.concat([df, row], ignore_index=True)
    df.index += 1
    st.dataframe(df)

st.subheader('📈 ROC Curve for Multi-Class Classification')

with st.echo():
    # Add diagonal line for random classifier (AUC = 0.5)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')

    # Customize the plot
    plt.title('ROC Curve for Multi-Class Classification')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True)
    plt.show()
st.pyplot(plt.gcf())

st.header('🚀 Conclusion')
st.markdown('''
    - KNN has the highest average AUC (0.54), meaning it has the best ability to distinguish between classes.
    - Random Forest has the best F1 score (0.46), meaning it balances precision and recall well.
    - Logistic Regression has the highest precision (0.62) but the worst AUC.
    - KNN has the highest AUC but the worst accuracy (0.46) and slowest inference time (1.77 ms).
    - Perceptron has decent AUC and is the fastest model, but lower F1 Score.
    ### Final Model Choice:
        1. Best Overall Model: Random Forest (Balanced Performance)
            - Good AUC, F1 Score, and Accuracy.
            - Faster than KNN
        2. Best If AUC is the Main Priority: KNN (Highest AUC)
''')
