# ELEN4025_Group_Assignment
Build and compare machine learning pipelines on the Open University Learning Analytics Dataset (OULAD). Provide Early at-risk prediction (binary) for the university students based on this data. "Early" means you use the students' data only up to an early week cutoff i.e. week 2, 4, etc.

## Task 2 cutoff handoff

Gift's Task 2 script builds row-level VLE handoff files for weeks 2, 4, 6, and 8:

```powershell
python scripts/build_task2_cutoffs.py
```

The generated CSVs are written to `data/interim/` and stay untracked because CSV outputs are ignored. See `docs/task2_handoff.md` for the handoff contract.
