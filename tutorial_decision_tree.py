"""A Decision Tree example to compare with the Logistic Regression lesson."""

import argparse
from pathlib import Path

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.data import load_dataset, split_features_target
from src.metrics import calculate_metrics


DATA_PATH = Path("data/wdbc.data")


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

    # A shallow tree is easier to inspect and helps limit overfitting.
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    malignant_probabilities = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, predictions, malignant_probabilities)

    print(f"Rows: {len(dataset)}; features: {features.shape[1]}")
    print(f"Train rows: {len(X_train)}; test rows: {len(X_test)}")
    print("Decision Tree metrics (malignant is the positive class):")
    for metric_name, score in metrics.items():
        print(f"  {metric_name}: {score:.4f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("This model does not need feature scaling.")


if __name__ == "__main__":
    main()