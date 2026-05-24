# OULAD Team Project Files Guide

This README explains the main files used for the OULAD group project and how the final notebook should be run.

## Main notebook to run

### `OULAD_Team_Notebook_Final_Integrated.ipynb`

This is the final integrated notebook for the group project.

It combines the main modelling and benchmarking work with the improved cross-validation, thresholding, ablation, audit, and reporting workflow.

The notebook includes:

1. Loading and checking the OULAD project files
2. Defining the target variable
3. Raw data audits
4. Missing-value, duplicate, row-count, and join-key checks
5. Registration inconsistency checks
6. Loading week 2, 4, 6, and 8 feature tables
7. Feature-level leakage checks
8. Advanced benchmark setup
9. Model training and comparison
10. Cross-validation workflow
11. Threshold tuning
12. Feature ablation and encoding ablation
13. Pipeline comparison tables
14. Best pipeline per week selection
15. Error analysis
16. Confusion matrix and ROC/AUC visualisations
17. Saving final tables and figures for the report

This is the notebook recommended for the final run because it contains the complete integrated workflow in one place.

## Source notebooks used during development

The following notebooks were used as development inputs before the final integrated notebook was created. They do not need to be run for the final output, but they explain where parts of the final workflow came from.

### `OULAD_Team_4.ipynb`

This notebook contained the original team modelling workflow, including the advanced benchmark, threshold visualisation, model comparison, error analysis, and performance plots.

### `task6_cv_ablation_new.ipynb`

This notebook focused on the improved cross-validation, thresholding, and ablation workflow. Its ideas were integrated into the final notebook so that the final workflow includes stronger evaluation and better report-ready tables.

## Why only one final notebook is recommended

The final integrated notebook is preferred because it reduces confusion and keeps the main workflow in one file. The earlier notebooks are useful for development history, but the final notebook should be used when generating the final results, tables, and figures.

