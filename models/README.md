## Pipeline 3 — Weighted Logistic Regression

Model file:
`pipeline_3_logreg_weighted_balanced.pkl`


Description:
This pipeline uses the shared group feature table and applies leakage-safe preprocessing using `ColumnTransformer`. Categorical variables are one-hot encoded and numerical variables are standardised. The classifier is logistic regression with `class_weight="balanced"`.

Purpose:
Baseline logistic regression pipeline with imbalance handling.