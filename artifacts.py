import joblib
from pathlib import Path


def save_everything_one_file(
    save_path: str,
    pipeline,
    report,
    metrics: dict,
    leaderboard_df,
    best_model_name: str,
    best_params: dict,
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,  # full sklearn pipeline (preprocess + model)
        "report": report.__dict__ if hasattr(report, "__dict__") else report,
        "metrics": metrics,
        "leaderboard": leaderboard_df.to_dict(orient="records") if leaderboard_df is not None else None,
        "best_model_name": best_model_name,
        "best_params": best_params,
    }

    joblib.dump(artifact, save_path)
    print(f"✅ Saved everything into one file: {save_path}")
