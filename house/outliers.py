import streamlit as st
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.mixture import GaussianMixture
import seaborn as sns
import math

st.title('🔍 Outlier Detection')

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

# Load data
df = pd.read_csv(CURRENT_DIR / 'Dataset1.csv')
categorical_features = df.select_dtypes(include=["object"]).columns
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

st.subheader('🔧 Train a Gaussian Mixture Model (GMM) on Dataset1 to identify potential outliers')
with st.echo():
    # n_components=2: assume two distributions: normal data and outliers.
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(df_encoded)

    # Compute log probabilities
    log_probs = gmm.score_samples(df_encoded)

    # Set threshold: The lower 5% as outliers
    threshold = np.percentile(log_probs, 5)

    # Label outliers (1 = outlier, 0 = normal)
    df_encoded["Outlier"] = (log_probs < threshold).astype(int)

    # Remove the detected outliers and save the cleaned dataset.
    df_cleaned = df_encoded[df_encoded["Outlier"] == 0].drop(columns=["Outlier"])
    df_cleaned.to_csv("house/Dataset1_Cleaned.csv", index=False)

    num_outliers = df_encoded["Outlier"].sum()

with st.container(border=True):
    st.write(f"Number of outliers: {num_outliers}")

st.subheader('📊 Visualize the histogram plot of the remaining data')
with st.echo():
    numeric_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns
    df_melted = df_cleaned[numeric_cols].melt(var_name="Feature", value_name="Value")
    g = sns.FacetGrid(df_melted, col="Feature", col_wrap=4, height=3, aspect=1.5, sharex=False, sharey=False)

    g.map_dataframe(sns.histplot, bins=30, kde=True, color="skyblue")

    g.fig.tight_layout()
st.pyplot(plt.gcf())

    