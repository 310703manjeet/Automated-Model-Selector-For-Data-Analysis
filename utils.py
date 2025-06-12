import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    st.write(f"🧩 Confusion Matrix for {model_name}")
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap="blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white')
    st.pyplot(fig)
