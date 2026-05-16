"""Train a calibrated CatBoost classifier on Week-4 OULAD cutoff data.

Reads:
    data/raw/studentInfo.csv
    data/interim/studentVle_week4.csv   (Task 2 handoff — already joined with activity_type)

Writes:
    models/pipeline_5_catboost.pkl
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
MODELS_DIR = REPO_ROOT / "models"

RANDOM_STATE = 42

CATEGORICAL_COLUMNS = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]

EXPECTED_ACTIVITY_TYPES = [
    "forumng",
    "homepage",
    "oucontent",
    "quiz",
    "resource",
    "subpage",
    "url",
    "page",
]


def load_student_info() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_RAW / "studentInfo.csv",
        dtype={
            "code_module": "string",
            "code_presentation": "string",
            "id_student": "Int64",
            "gender": "string",
            "region": "string",
            "highest_education": "string",
            "imd_band": "string",
            "age_band": "string",
            "num_of_prev_attempts": "Int64",
            "studied_credits": "Int64",
            "disability": "string",
            "final_result": "string",
        },
    )
    df["target"] = np.where(
        df["final_result"].isin(["Pass", "Distinction"]), 1, 0
    ).astype(int)
    return df


def build_vle_features() -> pd.DataFrame:
    """Aggregate Week-4 VLE clicks from the Task 2 handoff file."""
    vle = pd.read_csv(
        DATA_INTERIM / "studentVle_week4.csv",
        dtype={
            "code_module": "string",
            "code_presentation": "string",
            "id_student": "Int64",
            "id_site": "Int64",
            "date": "Int64",
            "sum_click": "Int64",
            "activity_type": "string",
        },
    )

    group_cols = ["code_module", "code_presentation", "id_student"]

    total_clicks = (
        vle.groupby(group_cols, as_index=False)["sum_click"]
        .sum()
        .rename(columns={"sum_click": "total_clicks_w4"})
    )
    active_days = (
        vle[group_cols + ["date"]]
        .drop_duplicates()
        .groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "active_days_w4"})
    )
    unique_sites = (
        vle[group_cols + ["id_site"]]
        .drop_duplicates()
        .groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "unique_sites_w4"})
    )

    activity_pivot = (
        vle.pivot_table(
            index=group_cols,
            columns="activity_type",
            values="sum_click",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    activity_pivot.columns = [str(c) for c in activity_pivot.columns]
    rename_map = {
        act: f"{act}_clicks_w4"
        for act in EXPECTED_ACTIVITY_TYPES
        if act in activity_pivot.columns
    }
    activity_pivot = activity_pivot.rename(columns=rename_map)
    for act in EXPECTED_ACTIVITY_TYPES:
        col = f"{act}_clicks_w4"
        if col not in activity_pivot.columns:
            activity_pivot[col] = 0

    features = total_clicks
    features = features.merge(active_days, on=group_cols, how="left")
    features = features.merge(unique_sites, on=group_cols, how="left")
    features = features.merge(activity_pivot, on=group_cols, how="left")

    features["clicks_per_active_day_w4"] = np.where(
        features["active_days_w4"].gt(0),
        features["total_clicks_w4"] / features["active_days_w4"],
        0.0,
    )
    return features


def prepare_for_catboost(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    frame = X.copy()
    for col in cat_cols:
        if col in frame.columns:
            frame[col] = frame[col].fillna("Missing").astype(str)
    return frame


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    student_info = load_student_info()
    vle_features = build_vle_features()

    group_cols = ["code_module", "code_presentation", "id_student"]
    feature_table = student_info.merge(vle_features, on=group_cols, how="left")

    exclude = {"id_student", "final_result", "target"}
    feature_columns = [c for c in feature_table.columns if c not in exclude]
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]

    X = feature_table[feature_columns].copy()
    y = feature_table["target"].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    X_train_core, X_calib, y_train_core, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )

    X_train_core_cb = prepare_for_catboost(X_train_core, cat_cols)
    X_calib_cb = prepare_for_catboost(X_calib, cat_cols)
    X_val_cb = prepare_for_catboost(X_val, cat_cols)

    print("Training CatBoost...")
    catboost = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=6,
        learning_rate=0.05,
        iterations=500,
        auto_class_weights="Balanced",
        random_state=RANDOM_STATE,
        verbose=False,
    )
    catboost.fit(
        X_train_core_cb,
        y_train_core,
        cat_features=cat_cols,
        eval_set=(X_calib_cb, y_calib),
        use_best_model=True,
        verbose=False,
    )

    print("Calibrating...")
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(catboost), method="sigmoid", cv=None
    )
    calibrated.fit(X_calib_cb, y_calib)

    val_auc = roc_auc_score(y_val, calibrated.predict_proba(X_val_cb)[:, 1])
    print(f"Validation AUC: {val_auc:.4f}")

    output_path = MODELS_DIR / "pipeline_5_catboost.pkl"
    joblib.dump(
        {
            "model": calibrated,
            "feature_columns": feature_columns,
            "cat_features": cat_cols,
        },
        output_path,
    )
    print(f"Saved: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
