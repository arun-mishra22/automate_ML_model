import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Splits df into:
      X_train, X_test, y_train, y_test

    Steps:
    ✅ separate X and y
    ✅ train test split
    ✅ stratify if classification
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    problem_type = problem_type.lower().strip()
    stratify = y if problem_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    return X_train, X_test, y_train, y_test
