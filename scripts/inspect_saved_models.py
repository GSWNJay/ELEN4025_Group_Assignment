import os
import joblib
import pandas as pd

MODEL_DIR = "models"
OUTPUT_PATH = "data/processed/model_inspection_summary.csv"

rows = []

print("\nINSPECTING SAVED MODEL FILES")
print("=" * 80)

for filename in sorted(os.listdir(MODEL_DIR)):
    if not filename.endswith(".pkl"):
        continue

    path = os.path.join(MODEL_DIR, filename)

    print("\n" + "=" * 80)
    print("File:", filename)
    print("Path:", path)

    row = {
        "file": filename,
        "loaded_successfully": False,
        "object_type": None,
        "is_pipeline": False,
        "pipeline_steps": None,
        "has_predict": False,
        "has_predict_proba": False,
        "has_decision_function": False,
        "classes": None,
        "n_features_in": None,
        "feature_names_in": None,
        "key_params": None,
        "load_error": None,
    }

    try:
        model = joblib.load(path)
        row["loaded_successfully"] = True
        row["object_type"] = str(type(model))

        print("Loaded successfully.")
        print("Object type:", type(model))

        row["has_predict"] = hasattr(model, "predict")
        row["has_predict_proba"] = hasattr(model, "predict_proba")
        row["has_decision_function"] = hasattr(model, "decision_function")

        print("Has predict:", row["has_predict"])
        print("Has predict_proba:", row["has_predict_proba"])
        print("Has decision_function:", row["has_decision_function"])

        if hasattr(model, "steps"):
            row["is_pipeline"] = True
            steps = []

            print("Pipeline steps:")
            for step_name, step_obj in model.steps:
                step_text = f"{step_name}: {type(step_obj).__name__}"
                steps.append(step_text)
                print("  -", step_text)

            row["pipeline_steps"] = " | ".join(steps)
        else:
            print("Not a sklearn Pipeline object.")

        if hasattr(model, "classes_"):
            row["classes"] = str(model.classes_)
            print("classes_:", model.classes_)

        if hasattr(model, "n_features_in_"):
            row["n_features_in"] = model.n_features_in_
            print("n_features_in_:", model.n_features_in_)

        if hasattr(model, "feature_names_in_"):
            names = list(model.feature_names_in_)
            row["feature_names_in"] = str(names)
            print("feature_names_in_ first 20:", names[:20])

        try:
            params = model.get_params()
            interesting = {}

            for key, value in params.items():
                key_lower = key.lower()
                if any(term in key_lower for term in [
                    "class_weight",
                    "penalty",
                    "solver",
                    "max_iter",
                    "random_state",
                    "n_neighbors",
                    "n_estimators",
                    "learning_rate",
                    "loss",
                    "calibration",
                    "estimator",
                    "base_estimator"
                ]):
                    interesting[key] = value

            row["key_params"] = str(interesting)
            print("Key params:", interesting)

        except Exception as e:
            print("Could not read parameters:", repr(e))

    except Exception as e:
        row["load_error"] = repr(e)
        print("LOAD ERROR:", repr(e))

    rows.append(row)

summary = pd.DataFrame(rows)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
summary.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 80)
print("Saved model inspection summary to:", OUTPUT_PATH)
print(summary)