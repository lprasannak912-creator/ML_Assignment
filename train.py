"""Train and save the five WDBC classifiers used for the assignment."""

import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data import load_dataset, split_features_target
from src.metrics import calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "wdbc.data"
MODEL_PATHS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib",
}


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=2,
            random_state=42,
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
        "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Support Vector Machine": make_pipeline(
            StandardScaler(),
            SVC(random_state=42),
        ),
    }


def save_test_set(X_test: pd.DataFrame, y_test: pd.Series) -> None:
    test_data = X_test.copy()
    test_data["diagnosis"] = y_test.map({0: "B", 1: "M"})
    output_path = PROJECT_ROOT / "data" / "test_data.csv"
    output_path.parent.mkdir(exist_ok=True)
    test_data.to_csv(output_path, index=False)
    print(f"Saved labeled hold-out set to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to the headerless UCI wdbc.data file.",
    )
    data_path = parser.parse_args().data

    dataset = load_dataset(data_path)
    features, target = split_features_target(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    save_test_set(X_test, y_test)

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    print(f"Rows: {len(dataset)}; features: {features.shape[1]}")
    print(f"Train rows: {len(X_train)}; test rows: {len(X_test)}")
    print("Diagnosis encoding: 0=benign, 1=malignant")

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            malignant_probabilities = model.predict_proba(X_test)[:, 1]
        else:
            malignant_probabilities = model.decision_function(X_test)
        metrics = calculate_metrics(y_test, predictions, malignant_probabilities)

        dump(model, model_dir / MODEL_PATHS[model_name])
        print(f"\n{model_name} metrics (malignant is the positive class):")
        for metric_name, score in metrics.items():
            print(f"  {metric_name}: {score:.4f}")
        print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()