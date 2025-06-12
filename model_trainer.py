import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_compare(models, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    trained_models = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append((name, acc))
            trained_models[name] = model

            st.write(f"✅ {name} Accuracy: **{acc:.4f}**")
        except Exception as e:
            st.warning(f"❌ {name} could not be trained: {e}")

    if results:
        st.subheader("📈 Accuracy Comparison")

        model_names = [r[0] for r in results]
        accuracies = [r[1] for r in results]

        fig, ax = plt.subplots()
        sns.barplot(x=accuracies, y=model_names, ax=ax, palette="viridis")
        ax.set_xlabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        st.pyplot(fig)

        best_model_name, best_acc = max(results, key=lambda x: x[1])
        st.success(f"🏆 Best Model: {best_model_name} with Accuracy = {best_acc:.4f}")

        return best_model_name, trained_models[best_model_name]
    else:
        st.error("No models could be trained. Please check your dataset.")
        return None, None
