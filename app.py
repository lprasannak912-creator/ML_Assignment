"""Streamlit app for evaluating saved WDBC classifiers on a labeled CSV upload."""

from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.metrics import confusion_matrix

from src.data import WDBC_FEATURES
from src.metrics import calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib",
}


def validate_test_data(test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required_columns = [*WDBC_FEATURES, "diagnosis"]
    missing_columns = [column for column in required_columns if column not in test_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if not test_data["diagnosis"].isin({"B", "M"}).all():
        raise ValueError("The diagnosis column must contain only B and M values.")

    feature_frame = test_data[WDBC_FEATURES].astype(float)
    target = test_data["diagnosis"].map({"B": 0, "M": 1})
    return feature_frame, target


st.set_page_config(page_title="WDBC Classifier Lab", layout="wide")
st.title("WDBC Classifier Lab")
st.caption("Educational model comparison. Not for medical diagnosis.")

model_name = st.selectbox(
    "Choose a saved classifier",
    [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Gaussian Naive Bayes",
        "Support Vector Machine",
    ],
)
uploaded_file = st.file_uploader(
    "Upload labeled test data (CSV)",
    type="csv",
    help="Include the 30 feature columns and a diagnosis column containing B or M.",
)

if uploaded_file is None:
    st.info("Upload a held-out test CSV to begin.")
else:
    model_path = PROJECT_ROOT / "models" / MODEL_FILES[model_name]
    if not model_path.exists():
        st.error(f"No saved model found at {model_path}. Run the training script first.")
        st.stop()

    try:
        test_data = pd.read_csv(uploaded_file)
        features, y_true = validate_test_data(test_data)
        model = load(model_path)
        predictions = model.predict(features)
        if hasattr(model, "predict_proba"):
            malignant_probabilities = model.predict_proba(features)[:, 1]
        else:
            malignant_probabilities = model.decision_function(features)
        metrics = calculate_metrics(y_true, predictions, malignant_probabilities)

        st.write(f"Selected model: {model_name}")
        st.write(f"Uploaded rows: {len(test_data)}")
        st.dataframe(test_data.head())

        metrics_df = pd.DataFrame(
            {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
        )
        st.subheader("Model metrics")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        confusion = confusion_matrix(y_true, predictions, labels=[0, 1])
        confusion_df = pd.DataFrame(
            confusion,
            index=["Actual B", "Actual M"],
            columns=["Predicted B", "Predicted M"],
        )
        st.subheader("Confusion matrix")
        st.dataframe(confusion_df, use_container_width=True)
    except ValueError as exc:
        st.error(str(exc))