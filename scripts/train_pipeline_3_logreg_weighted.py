import os
import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


def make_onehot_encoder():
    """
    Handles sklearn version differences
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Pipeline 3: weighted logistic regression."
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/week4_features.csv",
        help="Path to processed model-ready Week 4 feature table."
    )
    parser.add_argument(
        "--model-path",
        default="models/pipeline_3_logreg_weighted_balanced.pkl",
        help="Output path for saved pipeline."
    )
    parser.add_argument(
        "--target-col",
        default="label",
        help="Name of target column."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Could not find data file: {args.data_path}")

    df = pd.read_csv(args.data_path)

    if args.target_col not in df.columns:
        raise ValueError(
            f"Target column '{args.target_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    id_cols = ["id_student", "code_module", "code_presentation"]
    drop_cols = [args.target_col] + [col for col in id_cols if col in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[args.target_col]

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

    pipeline_3 = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline_3.fit(X_train, y_train)

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(pipeline_3, args.model_path)

    print("Saved model to:", args.model_path)
    print("Training data:", args.data_path)
    print("Target column:", args.target_col)
    print("Rows:", len(df))
    print("Features used:", X.shape[1])
    print("Categorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)


if __name__ == "__main__":
    main()