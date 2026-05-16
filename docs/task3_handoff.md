# Task 3 Feature Engineering Handoff

Surisha's Task 3 output builds leakage-safe week-cutoff feature tables from
Gift's Task 2 row-level VLE handoff files.

## How to Run

From the repository root:

```powershell
python scripts/build_task3_features.py
```

The script uses:

- `data/interim/studentVle_week2.csv`
- `data/interim/studentVle_week4.csv`
- `data/interim/studentVle_week6.csv`
- `data/interim/studentVle_week8.csv`
- `studentInfo.csv`, extracted from `anonymisedData.zip` into `data/raw/`

## Outputs

Task 3 writes generated files to `data/processed/`:

- `week2_features.csv`
- `week4_features.csv`
- `week6_features.csv`
- `week8_features.csv`
- `features_table.csv`
- `task3_summary.csv`

Each weekly feature table has one row per
`id_student`, `code_module`, and `code_presentation` record from
`studentInfo.csv` after binary label mapping.

## Feature Contract

The weekly tables contain:

- identifiers: `id_student`, `code_module`, `code_presentation`
- demographic features from `studentInfo.csv`
- binary target label where Pass/Distinction = 1 and Fail/Withdrawn = 0
- VLE totals up to the cutoff day:
  - `total_clicks_<week>`
  - `active_days_<week>`
  - high-frequency activity totals such as `quiz_clicks_<week>` and
    `forumng_clicks_<week>`

Missing demographic values are preserved for downstream modeling pipelines so
imputation can be fitted inside the train split. Missing VLE activity means no
observed activity and is encoded as 0.

## Validation

The build script verifies:

- Task 2 row counts for week 2, 4, 6, and 8
- each handoff file respects its cutoff day
- every VLE row has `activity_type`
- weekly feature tables have matching row counts
- labels are binary
- VLE feature columns are non-negative and non-missing
- no duplicate student/module/presentation rows exist
- activity totals do not exceed total clicks
- total clicks are monotonic from week 2 to week 8

The generated `features_table.csv` matches the assignment's Section 2.3
requirements: source CSVs, original columns, data type, definition, computation,
week availability, missing count, outlier count, duplicate count, leakage risk,
and notes.
