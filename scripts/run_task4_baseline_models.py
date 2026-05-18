import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)


def make_onehot_encoder():
    """
    Keeps the script compatible with different sklearn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_bagging_logreg():
    """
    Keeps the script compatible with different sklearn versions.
    """
    base_model = LogisticRegression(max_iter=2000, random_state=42)

    try:
        return BaggingClassifier(
            estimator=base_model,
            n_estimators=10,
            random_state=42
        )
    except TypeError:
        return BaggingClassifier(
            base_estimator=base_model,
            n_estimators=10,
            random_state=42
        )


def build_preprocessor(X):
    """
    Builds a leakage-safe preprocessing step.

    Categorical columns:
    - imputed using most frequent value
    - one-hot encoded

    Numeric columns:
    - imputed using median
    - standardised
    """
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder())
    ])

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numeric_pipeline, numeric_cols)
        ]
    )

    return preprocessor, categorical_cols, numeric_cols


def get_probability_scores(model, X_test):
    """
    Returns probability-like scores for class 1.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return 1 / (1 + np.exp(-scores))

    raise ValueError("Model has neither predict_proba nor decision_function.")


def evaluate_model(y_true, y_pred, y_proba):
    """
    Computes classification and probability-based metrics.
    Label 1 is Favourable and label 0 is Unfavourable.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_proba),
        "Log loss": log_loss(y_true, y_proba),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def main():
    week_files = {
        2: "data/processed/week2_features.csv",
        4: "data/processed/week4_features.csv",
        6: "data/processed/week6_features.csv",
        8: "data/processed/week8_features.csv",
    }

    baseline_models = {
        "B1_LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        "B2_L1_Regularised_LogisticRegression": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            random_state=42
        ),
        "B3_SGD_Logistic": SGDClassifier(
            loss="log_loss",
            max_iter=2000,
            tol=1e-3,
            random_state=42
        ),
        "B4_KNN": KNeighborsClassifier(
            n_neighbors=15
        ),
        "B5_GaussianNB": GaussianNB(),
        "B6_Bagging_LogisticRegression": make_bagging_logreg()
    }

    results = []

    for week, path in week_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing feature table: {path}")

        print("\n" + "=" * 80)
        print(f"Loading Week {week} feature table: {path}")

        df = pd.read_csv(path)

        target_col = "label"
        id_cols = ["id_student", "code_module", "code_presentation"]

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {path}")

        drop_cols = [target_col] + [col for col in id_cols if col in df.columns]

        X = df.drop(columns=drop_cols)
        y = df[target_col]

        preprocessor, categorical_cols, numeric_cols = build_preprocessor(X)

        print(f"Rows: {len(df)}")
        print(f"Features used: {X.shape[1]}")
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numeric columns: {numeric_cols}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        for model_name, model in baseline_models.items():
            print(f"Training {model_name} for Week {week}")

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            y_proba = get_probability_scores(pipeline, X_test)
            y_pred = (y_proba >= 0.5).astype(int)

            metrics = evaluate_model(y_test, y_pred, y_proba)

            results.append({
                "Run ID": f"W{week}_{model_name}",
                "Week cutoff": week,
                "Feature table": os.path.basename(path),
                "Model": model_name,
                "Class taught baseline": "Yes",
                "Self-learnt method": "No",
                "Preprocessing": "SimpleImputer + OneHotEncoder + StandardScaler",
                "Evaluation method": "80/20 stratified split",
                "Seed": 42,
                "Threshold": 0.5,
                "Number of rows": len(df),
                "Number of features": X.shape[1],
                "Feature columns used": ", ".join(X.columns.tolist()),
                **metrics
            })

    results_df = pd.DataFrame(results)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/task4_baseline_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print(f"Saved results to {output_path}")
    print(results_df)


if __name__ == "__main__":
    main()