import streamlit as st
from data_loader import load_data, preprocess_data
from model_selector import get_models
from model_trainer import train_and_compare
from evaluation import show_evaluation
from eda import run_eda
from feature_selector import select_features

def set_background_online():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://www.filepicker.io/api/file/OtDLNUgBTXu9uTO7MW5j");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def set_custom_style():
    st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        .main {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 15px;
        }

        h1, h2, h3 {
            color: white;
        }

        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def main():
    set_background_online()
    set_custom_style()

    st.markdown("<h1 style='text-align: center; color: white;'>🔮 AutoML Analyzer Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.title("📂 Upload Dataset")
        st.write("Upload a CSV file to begin.")
        df = load_data()

    if df is not None:
        st.sidebar.title("🔧 Feature Selection")
        df = select_features(df)

        X, y = preprocess_data(df)

        tabs = st.tabs(["📊 Dataset Overview", "🔍 EDA", "⚙️ Model Training", "🏆 Model Evaluation"])

        with tabs[0]:
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            st.write("📐 Shape:", df.shape)
            st.write("🔢 Class Distribution:")
            st.write(df.iloc[:, -1].value_counts())

        with tabs[1]:
            run_eda(df)

        with tabs[2]:
            models = get_models(X.shape[1])
            st.subheader("Compare Models by Accuracy")
            best_model_name, best_model = train_and_compare(models, X, y)
            st.session_state["best_model"] = best_model
            st.session_state["best_model_name"] = best_model_name

        with tabs[3]:
            st.subheader("Model Evaluation")
            if "best_model" in st.session_state:
                show_evaluation(st.session_state["best_model"], X, y, st.session_state["best_model_name"])
            else:
                st.info("Train a model first to view evaluation.")

if __name__ == "__main__":
    main()
