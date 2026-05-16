# Task 2 Cutoff Handoff

Gift's Task 2 output is a set of row-level VLE handoff files for Surisha's
Task 3 feature engineering. These files are filtered by week cutoff and already
include `activity_type` from `vle.csv`.

## How to Run

From the repository root:

```powershell
python scripts/build_task2_cutoffs.py
```

The script uses `anonymisedData.zip` from the repository root and extracts the
required raw CSVs into `data/raw/` when they are missing.

## Input File Contract

The raw OULAD file names are locked as:

- `data/raw/studentInfo.csv`
- `data/raw/studentVle.csv`
- `data/raw/vle.csv`

The join keys that must stay consistent with `studentInfo.csv` are
`id_student`, `code_module`, and `code_presentation`.

## Output File Contract

Task 2 writes generated files to `data/interim/`:

- `studentVle_week2.csv` where `date <= 13`
- `studentVle_week4.csv` where `date <= 27`
- `studentVle_week6.csv` where `date <= 41`
- `studentVle_week8.csv` where `date <= 55`

Each output has exactly these columns:

```text
code_module,code_presentation,id_student,id_site,date,sum_click,activity_type
```

The files remain row-level. They are not aggregated, imputed, encoded, scaled,
or joined to target labels. `data/processed/` belongs to Task 3.

## Validation Targets

The script validates that `date` is numeric, `sum_click` is preserved during
the join, every row receives `activity_type`, and the generated row counts are:

- Week 2: 1,757,383
- Week 4: 2,801,311
- Week 6: 3,562,771
- Week 8: 4,302,651

Negative dates are retained because the project brief defines each cutoff as
`date <= cutoff`.
