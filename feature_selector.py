import streamlit as st

def select_features(df):
    all_columns = df.columns.tolist()
    target_column = st.selectbox("🎯 Select Target Column", all_columns)

    feature_columns = st.multiselect("🧬 Select Feature Columns", [col for col in all_columns if col != target_column], default=[col for col in all_columns if col != target_column])

    return df[feature_columns + [target_column]]
