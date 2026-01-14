from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import pandas as pd


@dataclass
class RegistryInfo:
    problem_type: str
    imbalanced: bool
    class_distribution: Optional[dict]
    minority_ratio: Optional[float]


def _check_imbalance(y: pd.Series, threshold: float = 0.20) -> Tuple[bool, dict, float]:
    """
    Returns:
      imbalanced: bool
      distribution: dict (class -> ratio)
      minority_ratio: float
    """
    vc = y.value_counts(normalize=True)
    distribution = vc.to_dict()
    minority_ratio = float(vc.min()) if len(vc) > 0 else 1.0
    imbalanced = minority_ratio < threshold
    return imbalanced, distribution, minority_ratio


def get_model_registry(
    problem_type: str,
    y: Optional[pd.Series] = None,
    imbalance_threshold: float = 0.20
) -> Tuple[Dict[str, Any], RegistryInfo]:
    """
    Model Registry that also checks imbalance internally.

    Parameters
    ----------
    problem_type : str
        "classification" or "regression"
    y : pd.Series | None
        target series (required for imbalance check in classification)
    imbalance_threshold : float
        if minority_ratio < threshold -> imbalanced

    Returns
    -------
    models : Dict[str, sklearn model]
    info : RegistryInfo
    """
    problem_type = problem_type.lower().strip()

    # Defaults
    imbalanced = False
    class_distribution = None
    minority_ratio = None

    # If classification and y provided -> imbalance check
    if problem_type == "classification":
        if y is None:
            raise ValueError("For classification, 'y' is required to check imbalance inside registry.")

        imbalanced, class_distribution, minority_ratio = _check_imbalance(
            y=y, threshold=imbalance_threshold
        )

        class_weight = "balanced" if imbalanced else None

        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        models = {
            "logistic_regression": LogisticRegression(max_iter=2000, class_weight=class_weight),
            "random_forest": RandomForestClassifier(random_state=42, class_weight=class_weight),
            "svm": SVC(probability=True, random_state=42, class_weight=class_weight),
        }

        info = RegistryInfo(
            problem_type=problem_type,
            imbalanced=imbalanced,
            class_distribution=class_distribution,
            minority_ratio=minority_ratio
        )

        return models, info

    elif problem_type == "regression":
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR

        models = {
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(),
            "random_forest": RandomForestRegressor(random_state=42),
            "svr": SVR(),
        }

        info = RegistryInfo(
            problem_type=problem_type,
            imbalanced=False,
            class_distribution=None,
            minority_ratio=None
        )

        return models, info

    else:
        raise ValueError("Invalid problem type! Must be 'classification' or 'regression'.")
