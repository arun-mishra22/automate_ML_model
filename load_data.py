import pandas as pd
from pathlib import Path


def load_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)

    elif path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip().str.lower()
    return df


def split_features_target(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def dataset_summary(df: pd.DataFrame) -> dict:
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(df.isnull().sum().sum()),
        "missing_by_column": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return summary


def load_and_prepare_dataset(file_path: str, target_col: str):
    # Step 1: Load
    df_raw = load_dataset(file_path)

    # Step 2: Cleaning
    df_clean = basic_cleaning(df_raw)

    # Step 3: Dataset summary (after cleaning only)
    summary = dataset_summary(df_clean)

    # Step 4: Split X, y
    X, y = split_features_target(df_clean, target_col)

    return df_clean, X, y, summary

