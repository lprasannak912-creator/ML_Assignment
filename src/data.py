from pathlib import Path

import pandas as pd


WDBC_FEATURES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension",
]

WDBC_COLUMNS = ["id", "diagnosis", *WDBC_FEATURES]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the headerless UCI WDBC data file."""
    dataset = pd.read_csv(path, header=None, names=WDBC_COLUMNS)

    if dataset.shape[1] != 32:
        raise ValueError(f"Expected 32 columns; found {dataset.shape[1]}.")
    if dataset["diagnosis"].isna().any():
        raise ValueError("The diagnosis target contains missing values.")
    if not dataset["diagnosis"].isin({"B", "M"}).all():
        raise ValueError("Diagnosis values must be B or M.")
    if dataset[WDBC_FEATURES].isna().any().any():
        raise ValueError("One or more feature values are missing.")

    return dataset


def split_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Exclude the identifier and encode diagnosis as 0=benign, 1=malignant."""
    features = dataset[WDBC_FEATURES].astype(float)
    target = dataset["diagnosis"].map({"B": 0, "M": 1})
    return features, target