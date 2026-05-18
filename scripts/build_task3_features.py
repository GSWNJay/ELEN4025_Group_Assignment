"""Build Task 3 feature tables from the Task 2 OULAD cutoff handoff files.

The script consumes the row-level VLE cutoff CSVs in data/interim/, joins them
to cleaned demographics from studentInfo.csv, and writes leakage-safe feature
tables for week 2, 4, 6, and 8 into data/processed/.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


CUTOFFS = {"week2": 13, "week4": 27, "week6": 41, "week8": 55}
EXPECTED_INTERIM_ROWS = {
    "week2": 1_757_383,
    "week4": 2_801_311,
    "week6": 3_562_771,
    "week8": 4_302_651,
}
LABEL_MAP = {"Pass": 1, "Distinction": 1, "Fail": 0, "Withdrawn": 0}
CAT_COLS = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]
NUM_COLS = ["num_of_prev_attempts", "studied_credits"]
GROUP_KEYS = ["id_student", "code_module", "code_presentation"]

# High-frequency VLE activity types selected for per-type feature columns.
VLE_ACT_TYPES = [
    "oucontent",
    "quiz",
    "resource",
    "homepage",
    "subpage",
    "glossary",
    "oucollaborate",
    "forumng",
]

VLE_DTYPES = {
    "code_module": "category",
    "code_presentation": "category",
    "id_student": "int32",
    "id_site": "int32",
    "date": "int16",
    "sum_click": "int32",
    "activity_type": "category",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build Task 3 week-cutoff feature tables."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root containing anonymisedData.zip and data/interim/.",
    )
    parser.add_argument(
        "--skip-count-check",
        action="store_true",
        help="Skip validation against known Task 2 handoff row counts.",
    )
    return parser.parse_args()


def ensure_raw_files(repo_root: Path, file_names: tuple[str, ...]) -> Path:
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in file_names if not (raw_dir / name).exists()]
    if not missing:
        return raw_dir

    zip_path = repo_root / "anonymisedData.zip"
    if not zip_path.exists():
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing raw files ({missing_list}) and zip file was not found: {zip_path}"
        )

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_names = set(zip_ref.namelist())
        for file_name in missing:
            if file_name not in zip_names:
                raise FileNotFoundError(f"{file_name} is not present in {zip_path}")
            zip_ref.extract(file_name, raw_dir)
    print(f"Extracted raw files: {missing}")
    return raw_dir


def count_outliers_iqr(series: pd.Series) -> int:
    """Count values outside the 1.5 x IQR Tukey fence."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def prepare_demographics(student_info_raw: pd.DataFrame) -> pd.DataFrame:
    demo_cols = GROUP_KEYS + CAT_COLS + NUM_COLS + ["final_result"]
    student_info = student_info_raw[demo_cols].copy()
    student_info["label"] = student_info["final_result"].map(LABEL_MAP)
    student_info = student_info.dropna(subset=["label"])
    student_info["label"] = student_info["label"].astype(int)
    student_info = student_info.drop(columns=["final_result"])
    return student_info


def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[CAT_COLS] = cleaned[CAT_COLS].apply(
        lambda col: col.str.strip() if col.dtype == object else col
    )
    cleaned[CAT_COLS] = cleaned[CAT_COLS].replace({"": np.nan, "?": np.nan})
    cleaned["num_of_prev_attempts"] = cleaned["num_of_prev_attempts"].astype("Int64")
    cleaned["studied_credits"] = cleaned["studied_credits"].astype("Int64")
    return cleaned


def collect_activity_types(interim_dir: Path) -> set[str]:
    actual_types: set[str] = set()
    for name in CUTOFFS:
        path = interim_dir / f"studentVle_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing Task 2 handoff file: {path}")
        chunk = pd.read_csv(path, usecols=["activity_type"])
        actual_types.update(chunk["activity_type"].dropna().astype(str).unique())
    return actual_types


def aggregate_vle(
    df_vle: pd.DataFrame, cutoff: int, suffix: str, activity_types: list[str]
) -> pd.DataFrame:
    """Aggregate VLE interactions up to cutoff day inclusive."""
    filtered = df_vle[df_vle["date"] <= cutoff]

    total_clicks = (
        filtered.groupby(GROUP_KEYS, observed=True)["sum_click"]
        .sum()
        .rename(f"total_clicks_{suffix}")
    )
    active_days = (
        filtered.groupby(GROUP_KEYS, observed=True)["date"]
        .nunique()
        .rename(f"active_days_{suffix}")
    )
    act_series = []
    for activity_type in activity_types:
        activity_clicks = (
            filtered[filtered["activity_type"].astype(str) == activity_type]
            .groupby(GROUP_KEYS, observed=True)["sum_click"]
            .sum()
            .rename(f"{activity_type}_clicks_{suffix}")
            .reindex(total_clicks.index, fill_value=0)
        )
        act_series.append(activity_clicks)

    return pd.concat([total_clicks, active_days] + act_series, axis=1).reset_index()


def build_dataset(
    demographics: pd.DataFrame,
    student_vle: pd.DataFrame,
    cutoff: int,
    suffix: str,
    activity_types: list[str],
) -> pd.DataFrame:
    """Build one leakage-safe feature table for a week cutoff."""
    act_cols = [f"{a}_clicks_{suffix}" for a in activity_types]
    click_cols = [f"total_clicks_{suffix}", f"active_days_{suffix}"] + act_cols

    vle_agg = aggregate_vle(student_vle, cutoff, suffix, activity_types)
    df = demographics.merge(vle_agg, on=GROUP_KEYS, how="left")
    df[click_cols] = df[click_cols].fillna(0)
    df = clean_demographics(df)

    ordered = GROUP_KEYS + CAT_COLS + NUM_COLS + click_cols + ["label"]
    return df[ordered]


def validate_handoff(
    df_vle: pd.DataFrame,
    name: str,
    skip_count_check: bool,
) -> None:
    expected_rows = EXPECTED_INTERIM_ROWS[name]
    if not skip_count_check and len(df_vle) != expected_rows:
        raise ValueError(
            f"{name}: got {len(df_vle):,} handoff rows, expected {expected_rows:,}"
        )

    cutoff = CUTOFFS[name]
    max_date = int(df_vle["date"].max())
    if max_date > cutoff:
        raise ValueError(f"{name}: max date {max_date} exceeds cutoff {cutoff}")

    missing_activity = int(df_vle["activity_type"].isna().sum())
    if missing_activity:
        raise ValueError(f"{name}: {missing_activity:,} rows have no activity_type")


def validate_datasets(
    datasets: dict[str, pd.DataFrame], earliest_date: int, activity_types: list[str]
) -> None:
    all_passed = True
    baseline_len = len(datasets["week2"])

    print("\nSANITY CHECKS")
    for name, df in datasets.items():
        suffix = name
        cutoff_val = CUTOFFS[name]
        act_cols = [f"{a}_clicks_{suffix}" for a in activity_types]
        click_cols = [f"total_clicks_{suffix}", f"active_days_{suffix}"] + act_cols
        errors: list[str] = []

        vle_missing = int(df[click_cols].isnull().sum().sum())
        if vle_missing:
            errors.append(f"{vle_missing} missing values in VLE features")
        if set(df["label"].unique()) != {0, 1}:
            errors.append(f"unexpected labels: {set(df['label'].unique())}")
        for col in click_cols:
            if (df[col] < 0).any():
                errors.append(f"{col} has negative values")

        n_dupes = df.duplicated(subset=GROUP_KEYS).sum()
        if n_dupes:
            errors.append(f"{n_dupes} duplicate rows")

        for act_col in act_cols:
            bad = (df[act_col] > df[f"total_clicks_{suffix}"]).sum()
            if bad:
                errors.append(
                    f"{bad} rows where {act_col} exceeds total_clicks_{suffix}"
                )

        max_possible_days = cutoff_val - earliest_date + 1
        n_impossible = (df[f"active_days_{suffix}"] > max_possible_days).sum()
        if n_impossible:
            errors.append(f"{n_impossible} rows active_days > {max_possible_days}")
        if len(df) != baseline_len:
            errors.append(f"row count {len(df)} differs from week2 {baseline_len}")
        for col in NUM_COLS:
            if not pd.api.types.is_integer_dtype(df[col]):
                errors.append(f"{col} is not an integer type")

        status = "PASS" if not errors else "FAIL"
        if errors:
            all_passed = False
        print(f"[{status}] {name}: {', '.join(errors) if errors else 'all checks passed'}")

    print("\nMonotonicity checks")
    for w_early, w_late in zip(["week2", "week4", "week6"], ["week4", "week6", "week8"]):
        merged = datasets[w_early][GROUP_KEYS + [f"total_clicks_{w_early}"]].merge(
            datasets[w_late][GROUP_KEYS + [f"total_clicks_{w_late}"]],
            on=GROUP_KEYS,
        )
        violations = (
            merged[f"total_clicks_{w_late}"] < merged[f"total_clicks_{w_early}"]
        ).sum()
        status = "FAIL" if violations else "PASS"
        if violations:
            all_passed = False
        detail = f": {violations} violations" if violations else ""
        print(f"[{status}] {w_early} -> {w_late}{detail}")

    if not all_passed:
        raise ValueError("Task 3 sanity checks failed")


def build_features_table(
    datasets: dict[str, pd.DataFrame], activity_types: list[str]
) -> pd.DataFrame:
    ref_df = datasets["week4"]
    ref_suffix = "week4"
    records: list[dict[str, object]] = []

    demo_meta = [
        ("gender", "studentInfo.csv", "gender", "cat", "Student gender", "direct"),
        ("region", "studentInfo.csv", "region", "cat", "UK region of student", "direct"),
        (
            "highest_education",
            "studentInfo.csv",
            "highest_education",
            "cat",
            "Highest prior qualification",
            "direct",
        ),
        (
            "imd_band",
            "studentInfo.csv",
            "imd_band",
            "cat",
            "Index of multiple deprivation decile",
            "direct",
        ),
        ("age_band", "studentInfo.csv", "age_band", "cat", "Age band at registration", "direct"),
        (
            "disability",
            "studentInfo.csv",
            "disability",
            "cat",
            "Declared disability status",
            "direct",
        ),
        (
            "num_of_prev_attempts",
            "studentInfo.csv",
            "num_of_prev_attempts",
            "num",
            "Number of prior module attempts",
            "direct",
        ),
        (
            "studied_credits",
            "studentInfo.csv",
            "studied_credits",
            "num",
            "Total credits registered for",
            "direct",
        ),
    ]

    for feature, source, original, dtype, definition, how in demo_meta:
        col = ref_df[feature]
        records.append(
            {
                "Feature": feature,
                "Source CSV(s)": source,
                "Original column(s)": original,
                "Data type": dtype,
                "Definition": definition,
                "How computed": how,
                "Week availability": "2/4/6/8",
                "Missing count": int(col.isnull().sum()),
                "Outliers count": count_outliers_iqr(col) if dtype == "num" else "N/A",
                "Duplicate count": "0 (row-level, verified by sanity checks)",
                "Leakage risk": "None",
                "Notes": "Imputation deferred to modeling pipeline (structural NaN only)",
            }
        )

    vle_limitation = (
        "No-activity students get 0. Limitation: students who withdrew before "
        "cutoff appear as 0-click rows, not missing. studentRegistration not used."
    )
    vle_meta = [
        (
            "total_clicks",
            "studentVle.csv",
            "sum_click",
            "Total VLE clicks up to cutoff",
            "sum(sum_click) where date<=cutoff",
        ),
        (
            "active_days",
            "studentVle.csv",
            "date",
            "Distinct days with \u22651 click",
            "nunique(date) where date<=cutoff",
        ),
    ] + [
        (
            f"{activity_type}_clicks",
            "studentVle.csv + vle.csv",
            "sum_click, activity_type",
            f"Clicks on {activity_type} activity up to cutoff",
            f"sum where activity_type={activity_type}",
        )
        for activity_type in activity_types
    ]

    for feature, source, original, definition, how in vle_meta:
        col = ref_df[f"{feature}_{ref_suffix}"]
        records.append(
            {
                "Feature": feature,
                "Source CSV(s)": source,
                "Original column(s)": original,
                "Data type": "num",
                "Definition": definition,
                "How computed": how,
                "Week availability": "2/4/6/8",
                "Missing count": int(col.isnull().sum()),
                "Outliers count": count_outliers_iqr(col),
                "Duplicate count": "N/A",
                "Leakage risk": "None",
                "Notes": vle_limitation,
            }
        )

    return pd.DataFrame(records)


def build_summary_table(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, df in datasets.items():
        suffix = name
        n_features = len(
            [c for c in df.columns if c not in GROUP_KEYS + ["label"]]
        )
        rows.append(
            {
                "Dataset": name,
                "Cutoff (day)": CUTOFFS[name],
                "Rows": len(df),
                "Features": n_features,
                "Favourable (1)": int((df["label"] == 1).sum()),
                "Unfavourable (0)": int((df["label"] == 0).sum()),
                "Imbalance %": round((df["label"] == 0).sum() / len(df) * 100, 1),
                "Median total clicks": round(df[f"total_clicks_{suffix}"].median(), 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    raw_dir = ensure_raw_files(repo_root, ("studentInfo.csv",))
    interim_dir = repo_root / "data" / "interim"
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using repo root: {repo_root}")
    student_info_raw = pd.read_csv(raw_dir / "studentInfo.csv")
    student_info = prepare_demographics(student_info_raw)
    student_info = clean_demographics(student_info)
    print(f"studentInfo rows after label mapping: {len(student_info):,}")
    print(f"Label distribution: {student_info['label'].value_counts().to_dict()}")

    actual_types = collect_activity_types(interim_dir)
    activity_types = [a for a in VLE_ACT_TYPES if a in actual_types]
    missing_types = [a for a in VLE_ACT_TYPES if a not in actual_types]
    if missing_types:
        print(f"WARNING: activity types not found and excluded: {missing_types}")
    print(f"Activity features: {activity_types}")

    datasets: dict[str, pd.DataFrame] = {}
    earliest_date: int | None = None
    for name, cutoff in CUTOFFS.items():
        input_path = interim_dir / f"studentVle_{name}.csv"
        print(f"\nLoading {input_path.name}")
        student_vle = pd.read_csv(input_path, dtype=VLE_DTYPES)
        validate_handoff(student_vle, name, args.skip_count_check)
        earliest = int(student_vle["date"].min())
        earliest_date = earliest if earliest_date is None else min(earliest_date, earliest)

        print(f"Building {name} features (date <= {cutoff})")
        dataset = build_dataset(student_info, student_vle, cutoff, name, activity_types)
        datasets[name] = dataset

        output_path = processed_dir / f"{name}_features.csv"
        dataset.to_csv(output_path, index=False)
        print(f"  {dataset.shape} -> {output_path}")

    if earliest_date is None:
        raise ValueError("No Task 2 handoff data was loaded")
    validate_datasets(datasets, earliest_date, activity_types)

    features_table = build_features_table(datasets, activity_types)
    features_table_path = processed_dir / "features_table.csv"
    features_table.to_csv(features_table_path, index=False)
    print(f"\nFeatures table -> {features_table_path} ({len(features_table)} features)")

    summary_table = build_summary_table(datasets)
    summary_table_path = processed_dir / "task3_summary.csv"
    summary_table.to_csv(summary_table_path, index=False)
    print(f"Summary table -> {summary_table_path}")
    print(summary_table.to_string(index=False))

    print("\nTask 3 feature engineering completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
