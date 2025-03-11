import streamlit as st
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import io

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

# Load data
df = pd.read_csv(CURRENT_DIR / 'Dataset1.csv')

col1, col2, col3 = st.columns(3,border=True)

with col1:
    st.metric("Rows number", len(df))

with col2:
    st.metric("Columns number",len(df.columns))

with col3:
    st.metric("Missing value columns", (df.isnull().sum() > 0).sum())

st.subheader('🔬 Display basic statistics')

code = '''df.describe()'''
st.code(code, language="python")
st.write(df.describe())

st.subheader('🔍 Encode the categorical features (one-hot encoding)')
with st.echo():
    categorical_features = df.select_dtypes(include=["object"]).columns
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
buffer = io.StringIO()
df_encoded.info(buf=buffer)
with st.container(border=True):
    st.text(buffer.getvalue())

st.subheader('📊 The distribution of all features using histogram')
with st.echo():
    fig = df_encoded.hist(figsize=(12, 12), bins=30, edgecolor="black")
st.pyplot(plt.gcf())

st.subheader('📚 Original dataset with all features')
st.dataframe(df)



