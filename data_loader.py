import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data():
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        st.write("\U0001F4C4 Preview of Dataset:", df.head())
        return df
    return None

def preprocess_data(df):
    df = df.dropna(axis=0)  # Drop rows with missing values

    # Encoding
    y = df.iloc[:, -1]
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    y = np.ravel(y)

    X = df.iloc[:, :-1]
    X = pd.get_dummies(X)

    return X, y
