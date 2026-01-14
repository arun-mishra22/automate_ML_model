import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List


# -------------------------------
# Report (keep this for reporting)
# -------------------------------
@dataclass
class ValidationReport:
    problem_type: str
    dropped_columns: List[str]
    constant_columns: List[str]
    high_missing_columns: List[str]
    id_like_columns: List[str]
    leakage_columns: List[str]
    warnings: List[str]


# -------------------------------
# Helpers
# -------------------------------
def _missing_ratio(col: pd.Series) -> float:
    return float(col.isnull().mean())


def _unique_ratio(col: pd.Series) -> float:
    if len(col) == 0:
        return 0.0
    return float(col.nunique(dropna=True) / len(col))


def _is_constant(col: pd.Series) -> bool:
    return col.nunique(dropna=True) <= 1


def _detect_target_leakage_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """
    Leakage detection (NO heuristics):
    1) exact duplicate of target
    2) near-perfect correlation for numeric columns
    """
    LEAKAGE_CORR_THRESHOLD = 0.98

    leakage_cols = []
    y = df[target_col]

    # 1) exact match
    for col in df.columns:
        if col == target_col:
            continue
        try:
            if df[col].equals(y):
                leakage_cols.append(col)
        except Exception:
            pass

    # 2) correlation check (numeric only)
    if pd.api.types.is_numeric_dtype(y):
        num_df = df.select_dtypes(include=[np.number]).copy()
        num_df = num_df.drop(columns=[target_col], errors="ignore")

        if not num_df.empty:
            corr = num_df.corrwith(y).abs()
            corr_leaks = corr[corr >= LEAKAGE_CORR_THRESHOLD].index.tolist()
            leakage_cols.extend(corr_leaks)

    return sorted(list(set(leakage_cols)))


# -------------------------------
# Main functions
# -------------------------------
def validate_dataset(df: pd.DataFrame, target_col: str) -> None:
    MIN_ROWS = 20

    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.shape[0] < MIN_ROWS:
        raise ValueError(f"Dataset has too few rows (<{MIN_ROWS}).")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    if df[target_col].isnull().sum() > 0:
        raise ValueError("Target column contains missing values. Handle target missing values first.")

    if df[target_col].nunique(dropna=True) <= 1:
        raise ValueError("Target column has only 1 unique value (not trainable).")


def detect_problem_type(y: pd.Series) -> str:
    """
    Better problem-type detection:
    - object/category/bool => classification
    - integer with small unique values => classification
    - float with small unique values => classification
    - else regression
    """
    if y is None:
        raise ValueError("y cannot be None.")

    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"

    nunique = y.nunique(dropna=True)

    if pd.api.types.is_integer_dtype(y) and nunique <= 20:
        return "classification"

    if pd.api.types.is_numeric_dtype(y) and nunique <= 10:
        return "classification"

    return "regression"


def clean_and_validate(df: pd.DataFrame, target_col: str):
    """
    Default cleaning + validation:
    ✅ drop high missing cols (>60%)
    ✅ drop constant cols
    ✅ drop id-like cols (unique ratio >= 0.95)
    ✅ leakage detection (warn only)
    ✅ detect problem type
    """
    MISSING_THRESHOLD = 0.60
    ID_UNIQUE_RATIO = 0.95

    validate_dataset(df, target_col)

    df = df.copy()
    warnings = []

    dropped_columns = []
    constant_columns = []
    high_missing_columns = []
    id_like_columns = []
    leakage_columns = []

    # 1) Drop high missing columns
    for col in df.columns:
        if col == target_col:
            continue
        if _missing_ratio(df[col]) > MISSING_THRESHOLD:
            high_missing_columns.append(col)

    if high_missing_columns:
        df.drop(columns=high_missing_columns, inplace=True, errors="ignore")
        dropped_columns.extend(high_missing_columns)

    # 2) Drop constant columns
    for col in df.columns:
        if col == target_col:
            continue
        if _is_constant(df[col]):
            constant_columns.append(col)

    if constant_columns:
        df.drop(columns=constant_columns, inplace=True, errors="ignore")
        dropped_columns.extend(constant_columns)

    # 3) Detect & Drop ID-like columns
    for col in df.columns:
        if col == target_col:
            continue

        colname = col.lower()

        # common ID patterns
        if colname in ["id", "customerid", "userid"] or colname.endswith("_id"):
            id_like_columns.append(col)
            continue

        # unique ratio check
        ur = _unique_ratio(df[col])
        if ur >= ID_UNIQUE_RATIO:
            id_like_columns.append(col)

    if id_like_columns:
        df.drop(columns=id_like_columns, inplace=True, errors="ignore")
        dropped_columns.extend(id_like_columns)

    # 4) Leakage detection (warn only)
    leakage_columns = _detect_target_leakage_columns(df, target_col)
    if leakage_columns:
        warnings.append(
            f"Possible target leakage columns detected: {leakage_columns}. "
            f"Review them carefully (AutoML will NOT drop them automatically)."
        )

    # 5) problem type
    problem_type = detect_problem_type(df[target_col])

    report = ValidationReport(
        problem_type=problem_type,
        dropped_columns=sorted(list(set(dropped_columns))),
        constant_columns=sorted(list(set(constant_columns))),
        high_missing_columns=sorted(list(set(high_missing_columns))),
        id_like_columns=sorted(list(set(id_like_columns))),
        leakage_columns=sorted(list(set(leakage_columns))),
        warnings=warnings,
    )

    return df, report
