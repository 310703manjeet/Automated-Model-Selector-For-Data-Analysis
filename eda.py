import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    st.subheader("📈 Exploratory Data Analysis")

    if df is not None:
        st.write("### 🔍 Basic Info")
        st.write(df.describe())
        st.write("### 🧩 Missing Values")
        st.write(df.isnull().sum())

        st.write("### 📊 Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="viridis", ax=ax)
        st.pyplot(fig)

        st.write("### 📉 Distribution Plots")
        num_cols = df.select_dtypes(include='number').columns
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
