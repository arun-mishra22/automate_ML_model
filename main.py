import joblib
import pandas as pd

# 1) load
from load_data import load_dataset

# 2) validation + cleaning
from data_validation import clean_and_validate

# 3) preprocessing
from data_preprocessing import build_preprocessor   # you should have this function

# 4) split
from train_test_split import split_data                   # you should have this function

# 5) registry
from model_registry import get_model_registry

# 6) train
from model_trainer import train_and_select_best_model

# 7) tuning
from hyperparameter_tunning import tune_best_model

# 8) evaluation
from evaluation import evaluate_model

# 9) artifact saving (one-file)
from artifacts import save_everything_one_file


def main():
    print("\n==============================")
    print("✅ AutoML Pipeline STARTED")
    print("==============================\n")

    file_path = "dataset.csv"
    target_col = "Churn"

    # ---------------------------
    # STEP 1: Load
    # ---------------------------
    print(" 1) Loading dataset...")
    df = load_dataset(file_path)
    print("   df shape:", df.shape)

    # ---------------------------
    # STEP 2: Clean + Validate
    # ---------------------------
    print("\n 2)  Cleaning + validating...")
    df_clean, report = clean_and_validate(df, target_col)
    print("   Clean df shape:", df_clean.shape)
    print("   Problem type:", report.problem_type)


    # ---------------------------
    # STEP 3: Train/Test Split
    # ---------------------------
    print("\n3️⃣ Train-Test split...")
    X_train, X_test, y_train, y_test = split_data(
        df_clean,
        target_col,
        report.problem_type
    )
    print("   Train shape:", X_train.shape, y_train.shape)
    print("   Test shape :", X_test.shape, y_test.shape)

    # ---------------------------
    # STEP 5: Preprocessor
    # ---------------------------
    print("\n4️⃣ Building preprocessor...")
    preprocessor = build_preprocessor(X_train)
    print("   Preprocessor built ✅")

    # ---------------------------
    # STEP 6: Model Registry (+ imbalance)
    # ---------------------------
    print("\n5️⃣ Getting model registry...")
    models, reg_info = get_model_registry(
        problem_type=report.problem_type,
        y=y_train if report.problem_type == "classification" else None
    )
    print("   Models:", list(models.keys()))
    if report.problem_type == "classification":
        print("   Imbalanced:", reg_info.imbalanced)
        print("   Distribution:", reg_info.class_distribution)

    # ---------------------------
    # STEP 7: Train models
    # ---------------------------
    print("\n6️⃣ Training & selecting best model...")
    best_model_name, best_pipeline, leaderboard_df = train_and_select_best_model(
        X=X_train,
        y=y_train,
        preprocessor=preprocessor,
        models=models,
        problem_type=report.problem_type,
        cv=5
    )
    print("   Best model:", best_model_name)
    print("   Leaderboard top 3:\n", leaderboard_df.head(3))

    # ---------------------------
    # STEP 8: Hyperparameter tuning
    # ---------------------------
    print("\n  Hyperparameter tuning...")
    tuned_pipeline, best_params, best_cv_score = tune_best_model(
        best_pipeline=best_pipeline,
        best_model_name=best_model_name,
        X=X_train,
        y=y_train,
        problem_type=report.problem_type,
        cv=5
    )
    print("   Best params:", best_params)
    print("   Best CV score:", best_cv_score)

    # ---------------------------
    # STEP 9: Evaluation
    # ---------------------------
    print("\n8️⃣ Evaluating tuned pipeline...")
    test_metrics = evaluate_model(
        pipeline=tuned_pipeline,
        X_test=X_test,
        y_test=y_test,
        problem_type=report.problem_type
    )
    print("   Test metrics:", test_metrics)

    # ---------------------------
    # STEP 10: Save final artifact
    # ---------------------------
    print("\n9️⃣ Saving artifact...")
    save_everything_one_file(
        save_path="artifacts/automl_artifact.joblib",
        pipeline=tuned_pipeline,
        report=report,
        metrics=test_metrics,
        leaderboard_df=leaderboard_df,
        best_model_name=best_model_name,
        best_params=best_params,
    )

    print("\n==============================")
    print("🎉 AutoML Pipeline COMPLETED")
    print("==============================\n")


if __name__ == "__main__":
    main()
