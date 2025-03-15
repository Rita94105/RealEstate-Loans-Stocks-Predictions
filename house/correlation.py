import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

st.title('✅ Correlation and Feature Selection')

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
# Load cleaned data
df_cleaned = pd.read_csv(CURRENT_DIR / 'Dataset1_Cleaned.csv')

st.subheader('📊 Visualize correlations between features and target')
with st.echo():
    # Compute correlation matrix
    corr_matrix = df_cleaned.corr()

    # Set up the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1)

    # Title
    plt.title("Feature Correlation Heatmap", fontsize=14)
st.pyplot(plt.gcf())

with st.echo():
    target_corr = corr_matrix["House_Price"].drop("House_Price").sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    sns.heatmap(target_corr.to_frame(), annot=True, cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1)

    plt.title("Feature-Target Correlation", fontsize=14)
st.pyplot(plt.gcf())

st.subheader('🔍 Features with correlation > 0.02')
with st.echo():
    features_above_002 = target_corr[target_corr > 0.02].index.to_list()
with st.container(border=True):
    st.table(features_above_002)

st.subheader('🔍 Features with correlation > 0.04')
with st.echo():
    features_above_004 = target_corr[target_corr > 0.04].index.to_list()
with st.container(border=True):
    st.table(features_above_004)




