# Legacy project map

Use this reference only when migrating or interpreting the code under the project root.

## Pipeline map

| Stage | Legacy location | Behavior to retain | Migration requirement |
|---|---|---|---|
| EDA entry | `数据源分析EDA/main.py` | Config-driven CSV to Excel EDA report | Keep separate from training; share one field-role configuration. |
| EDA analytics | `数据源分析EDA/eda/analytics/` | IV/KS, chi-bin, PSI, correlation, monthly checks, plots | Preserve metric calculations; validate date and categorical-variable handling. |
| Data cleaning | `风控建模RM/src/data/data_clean.py` | Date normalization, high-missing and high-constant filters | Apply immutable, logged transforms; parameterize the 0.8 thresholds. |
| Feature policy | `风控建模RM/src/config/feature_config.py` | Excluded columns | Make exclusions run-configured; do not mutate module-level lists. |
| Split | `风控建模RM/src/models/model_tools.py:sample_select` | Reserve OOT months, stratified train/test | Reserve OOT before tuning; configure `test_size`, seed, date column, and OOT months. |
| Training | `风控建模RM/src/models/lightgbm_model.py` | LightGBM and parameter search | Train on train only and select on test. Keep OOT out of selection. |
| Feature selection | `风控建模RM/src/models/model_tools.py:model_vars_imp` | Importance-based variable reduction | Record the first model, rule, retained fields, and rerun result. |
| Evaluation | `风控建模RM/src/models/model_tran_utlis.py` | KS/AUC/PSI by sample split | Return numeric metrics rather than formatted strings. Avoid mutating input frames. |
| Report | `风控建模RM/src/models/model_report.py` | Excel, plots, stability, PMML attempt | Make report generation optional and resilient to PMML/Office availability. |

## Legacy execution sequence

1. A Notebook reads a CSV and normalizes its date.
2. It removes high-missing and high-constant fields, then constructs a feature list by exclusion.
3. It reserves hard-coded OOT months and makes a stratified random train/test split.
4. It runs LightGBM Bayesian parameter search, optionally reduces features by importance, and retrains.
5. It writes a pickle, scores the cleaned input, and generates an Excel evaluation report.

## Migration hazards

- EDA and model training normalize dates independently and use different month-column names.
- Notebooks include hard-coded input paths, OOT months, and interactive model indexes.
- The legacy target/test results can influence parameter selection; protect OOT as a final holdout.
- `NOT_MODEL_TRAIN_FEATURE` is mutated in a Notebook; make per-run copies.
- Model scoring appends prediction columns to the input DataFrame; operate on a copy.
- There is no dependency manifest. Capture runtime versions and optional dependencies such as `lightgbm`, `toad`, `openpyxl`, and PMML libraries in the new project.
