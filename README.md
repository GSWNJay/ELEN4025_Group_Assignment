# ELEN4025_Group_Assignment
Build and compare machine learning pipelines on the Open University Learning Analytics Dataset (OULAD). Provide Early at-risk prediction (binary) for the university students based on this data. "Early" means you use the students' data only up to an early week cutoff i.e. week 2, 4, etc.

## Optional Environment Setup (Virtual Environment)
To ensure all scripts and the main notebook run correctly without dependency conflicts, please set up a Python virtual environment before running any code.

**1. Create the virtual environment:**
Run this command in the root directory of the project:
```bash
python -m venv venv
```

**2. Activate the virtual environment:**

- Windows (Command Prompt): venv\Scripts\activate.bat

- Windows (PowerShell): venv\Scripts\Activate.ps1

- macOS / Linux: source venv/bin/activate

**3. Install dependencies:**
Install all required libraries (like catboost, xgboost, scikit-learn, etc.) using the requirements file:

```Bash
pip install -r requirements.txt
```

**4. Link the environment to Jupyter Notebook:**
Since the final evaluation is done in a Jupyter Notebook, register this virtual environment as a kernel:

```Bash
python -m ipykernel install --user --name=oulad_env --display-name "Python (OULAD Group Project)"
```
(Note: When you open OULAD_Team_4.ipynb, make sure to select "Python (OULAD Group Project)" as your active kernel!)


## Task 2 cutoff handoff

Gift's Task 2 script builds row-level VLE handoff files for weeks 2, 4, 6, and 8:

```powershell
python scripts/build_task2_cutoffs.py
```

The generated CSVs are written to `data/interim/` and stay untracked because CSV outputs are ignored. See `docs/task2_handoff.md` for the handoff contract.

## Task 3 feature engineering

Surisha's Task 3 script builds leakage-safe processed feature tables from the
Task 2 handoff files:

```powershell
python scripts/build_task3_features.py
```

The generated week 2/4/6/8 feature tables and assignment Features Table are
written to `data/processed/`. See `docs/task3_handoff.md` for the output and
validation contract.

