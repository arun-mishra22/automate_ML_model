import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


def get_param_distributions(best_model_name: str):
    """
    Returns param distributions (search space) based on model name.
    The keys use sklearn Pipeline format: model__param
    """
    name = best_model_name.lower().strip()

    # ------------------ CLASSIFICATION ------------------
    if name == "logistic_regression":
        return {
            "model__C": np.logspace(-3, 2, 20),
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
        }

    if name == "svm":  # SVC
        return {
            "model__C": np.logspace(-2, 2, 15),
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        }

    if name == "random_forest":
        return {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 3, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        }

    # ------------------ REGRESSION ------------------
    if name == "ridge_regression" or name == "ridge":
        return {
            "model__alpha": np.logspace(-4, 2, 30)
        }

    if name == "lasso":
        return {
            "model__alpha": np.logspace(-4, 1, 30)
        }

    if name == "svr":  # Support Vector Regression
        return {
            "model__C": np.logspace(-2, 2, 15),
            "model__epsilon": np.linspace(0.01, 0.5, 10),
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        }

    # no tuning for linear regression
    if name == "linear_regression":
        return None

    # if unknown model name
    return None


def tune_best_model(
    best_pipeline: Pipeline,
    best_model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    cv: int = 5,
    n_iter: int = 30,
    random_state: int = 42
):
    """
    Tunes best_pipeline using RandomizedSearchCV.
    Returns:
      tuned_pipeline, best_params, best_cv_score

    If model has no tuning space, returns original pipeline.
    """

    problem_type = problem_type.lower().strip()

    # pick scoring
    if problem_type == "classification":
        scoring = "f1_weighted"
    elif problem_type == "regression":
        scoring = "r2"
    else:
        raise ValueError("problem_type must be 'classification' or 'regression'")

    param_distributions = get_param_distributions(best_model_name)

    # If no tuning space → skip tuning
    if param_distributions is None:
        # fit on full data (to be safe)
        best_pipeline.fit(X, y)
        return best_pipeline, {}, None

    search = RandomizedSearchCV(
        estimator=best_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    search.fit(X, y)

    tuned_pipeline = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    # Fit tuned model on full dataset (important)
    tuned_pipeline.fit(X, y)

    return tuned_pipeline, best_params, best_score
