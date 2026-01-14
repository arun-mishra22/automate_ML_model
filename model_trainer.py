import time
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold


def train_and_select_best_model(
    X,
    y,
    preprocessor,
    models: dict,
    problem_type: str,
    cv: int = 5,
    scoring: str | None = None
):
    # choose scoring automatically if not provided
    if scoring is None:
        if problem_type == "classification":
            scoring = "f1_weighted"
        elif problem_type == "regression":
            scoring = "r2"
        else:
            raise ValueError("problem_type must be 'classification' or 'regression'")

    # choose CV splitter
    if problem_type == "classification":
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

    results = []
    best_score = float("-inf")
    best_model_name = None
    best_pipeline = None

    for model_name, model in models.items():
        start = time.time()

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        try:
            scores = cross_val_score(pipe, X, y, cv=cv_splitter, scoring=scoring)
            mean_score = float(scores.mean())
            std_score = float(scores.std())

            fit_time = time.time() - start

            results.append({
                "model": model_name,
                "scoring": scoring,
                "cv_mean_score": mean_score,
                "cv_std_score": std_score,
                "fit_time_sec": fit_time
            })

            if mean_score > best_score:
                best_score = mean_score
                best_model_name = model_name
                best_pipeline = pipe

        except Exception as e:
            results.append({
                "model": model_name,
                "scoring": scoring,
                "cv_mean_score": None,
                "cv_std_score": None,
                "fit_time_sec": None,
                "error": str(e)
            })

    leaderboard_df = pd.DataFrame(results).sort_values(
        by="cv_mean_score",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    if best_pipeline is None:
        raise RuntimeError("No model could be trained successfully. Check dataset and preprocessing.")

    # final fit
    best_pipeline.fit(X, y)

    return best_model_name, best_pipeline, leaderboard_df
