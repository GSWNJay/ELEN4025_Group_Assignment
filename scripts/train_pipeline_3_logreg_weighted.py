import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# Pipeline 3: Weighted Logistic Regression
# Contributor: Inathi

DATA_PATH = "data/model_data_w4.csv"

MODEL_PATH = "models/pipeline_3_logreg_weighted_balanced.pkl"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. "
        "Please update DATA_PATH to the final shared feature-table path."
    )

model_data = pd.read_csv(DATA_PATH)

drop_cols = [
    "target",
    "final_result",
    "id_student",
    "code_module",
    "code_presentation"
]

drop_cols = [col for col in drop_cols if col in model_data.columns]

X = model_data.drop(columns=drop_cols)
y = model_data["target"]

categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
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

pipeline_3.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline_3, f)

print(f"Saved model to: {MODEL_PATH}")
print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)
print("Training rows:", X_train.shape[0])
print("Test rows:", X_test.shape[0])