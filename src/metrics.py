from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, predictions, malignant_probabilities):
    """Return the assignment metrics, treating malignant (1) as positive."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "auc": roc_auc_score(y_true, malignant_probabilities),
        "precision": precision_score(y_true, predictions, pos_label=1),
        "recall": recall_score(y_true, predictions, pos_label=1),
        "f1": f1_score(y_true, predictions, pos_label=1),
        "mcc": matthews_corrcoef(y_true, predictions),
    }