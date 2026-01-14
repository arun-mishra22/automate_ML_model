import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> dict:
    """
    Evaluate trained pipeline on a holdout test set.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        trained full pipeline (preprocess + model)
    X_test : pd.DataFrame
        test features
    y_test : pd.Series
        test target
    problem_type : str
        "classification" or "regression"

    Returns
    -------
    dict : metrics
    """

    problem_type = problem_type.lower().strip()

    # Predictions
    y_pred = pipeline.predict(X_test)

    # -------------------------
    # CLASSIFICATION
    # -------------------------
    if problem_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, zero_division=0)
        }
        return metrics

    # -------------------------
    # REGRESSION
    # -------------------------
    elif problem_type == "regression":
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(rmse),
        }
        return metrics

    else:
        raise ValueError("problem_type must be 'classification' or 'regression'")
