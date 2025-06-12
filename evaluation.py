import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def show_evaluation(model, X, y_true, model_name):
    y_pred = model.predict(X)

    st.write(f"### 📌 Evaluation for: {model_name}")

    st.subheader("📄 Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="Blues"))

    st.subheader("🧮 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("🎯 Sample Predictions")
    st.write(f"Actual: {y_true[:5].tolist()}")
    st.write(f"Predicted: {y_pred[:5].tolist()}")
