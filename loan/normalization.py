import streamlit as st
import pandas as pd
import pathlib

st.title('📚 Data Splitting and Normalization')

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

# Load data
df = pd.read_csv(CURRENT_DIR / 'Dataset2.csv')
categorical_features = df.select_dtypes(include=["object"]).columns
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

st.subheader('✂️ Split the data into train, validation and test')

code='''
    X2 = df_encoded.drop(columns=['Loan_Status'])
    y2 = df_encoded['Loan_Status']
    # Split the data into train, validation and test and hold out 20% and 10% of observations as the validation and test set
    X2_train, X2_temp, y2_train, y2_temp = train_test_split(X2, y2, test_size=0.3, random_state=42)
    X2_val, X2_test, y2_val, y2_test = train_test_split(X2_temp, y2_temp, test_size=1/3, random_state=42)'''
st.code(code, language="python")

st.subheader('🔧 Normalize the data (Z-normalization)')

code='''
    # Normalize Data (Z-score normalization)
    scaler = StandardScaler()
    X2_train = scaler.fit_transform(X2_train)
    X2_val = scaler.transform(X2_val)
    X2_test = scaler.transform(X2_test)' \
    '''
st.code(code, language="python")
