# Configuration reference

## Data contract

Use `assets/data_contract.template.yaml` as the starting point.

- `data.input_path`: source path relative to the contract file, or an absolute path.
- `schema.id_cols`: one or more columns that identify a record; combined uniqueness must be at least 95% under the current validator.
- `schema.date_col`: observation date used to derive `event_date` and `event_month`.
- `schema.target.column`: confirmed binary target.
- `schema.target.good_label` / `bad_label`: explicit target mapping. Default risk convention is good=`0`, bad=`1` only when confirmed.
- `schema.exclude_cols`: post-outcome, manually excluded, operational, or otherwise leakage-prone fields.

The loader accepts CSV, Excel, and Parquet. Confirm delimiter and encoding when a CSV does not parse normally.

## Sample configuration

The standalone sample node uses `assets/sample_diagnosis.template.yaml` as its
single configuration. Diagnostic thresholds classify evidence but never change
data; treatment actions are hash-bound user decisions recorded after approval
and applied only by a later preprocessing/modeling node. The former
`sample_config.yaml` shape remains read-only compatibility for legacy batch
runs.

- `diagnostics.imbalance_*_minority_rate`: warning and critical thresholds for the minority-class share.
- `diagnostics.latest_month_min_volume_ratio`: compare the latest month with the median of the preceding three months.
- `diagnostics.monthly_bad_rate_change_warning`: absolute adjacent-month bad-rate change that triggers review.
- `diagnostics.high_missing_row_rate`: candidate-feature missingness threshold for row-level review.
- `treatment.duplicate_action`: `error`, `keep_first`, or `keep_last`.
- `treatment.missing_target_action`: `error` or `drop`.
- `treatment.all_null_feature_action`: `error` or `drop`.
- `treatment.high_missing_row_action`: `keep` or `drop`.
- `treatment.incomplete_latest_month_action`: `error`, `keep`, or `exclude`.
- `treatment.class_imbalance_action`: `none` or Train-only `class_weight`.

Do not use over/under-sampling on Test or OOT. This version deliberately supports class weighting before synthetic sampling because it preserves rows and is easier to audit.

## Model configuration

Use `assets/model_config.template.yaml` as the starting point.

- `split.oot_months`: reserve the newest distinct months before any Train/Test work.
- `split.test_ratio`: stratified fraction taken from the remaining development sample.
- `split.random_seed`: make the split and LightGBM sampling reproducible.
- `feature_preprocessing.numeric.missing_strategy`: currently `native`; retain nulls for LightGBM instead of learning an imputation value from Test/OOT.
- `feature_preprocessing.categorical.strategy`: `lightgbm_native` or conservative `drop`.
- `feature_preprocessing.categorical.rare_*`: Train-only low-frequency grouping policy.
- `feature_preprocessing.categorical.unknown_action`: map Test/OOT categories absent from Train to missing or the Train-derived `OTHER` group.
- `feature_preprocessing.categorical.near_unique_rate`: exclude ID-like categorical features.
- `feature_preprocessing.categorical.max_categories` / `high_cardinality_action`: bound native categorical complexity or drop high-cardinality fields.
- `feature_preprocessing.text.*`: detect and drop free-text-like fields in this structured-data workflow.
- `feature_selection.min_iv`: remove weak univariate features using Train-only IV.
- `feature_selection.max_correlation`: among highly correlated numeric pairs, retain the more stable, higher-IV, lower-missing representative.
- `feature_selection.max_missing_rate`: final Train-only missingness guardrail.
- `feature_selection.max_dominant_rate_warning`: flag sparse or highly concentrated features for review without automatically deleting them.
- `feature_selection.max_psi` / `max_unstable_month_ratio`: identify repeated Train-month distribution instability; `stability_action` controls review versus deletion.
- `feature_selection.min_month_samples`: ignore unsupported months in automated stability decisions.
- `feature_selection.correlation_method`: `spearman` is the default; `pearson` remains available.
- `training.mode`: use `baseline` for one configured LightGBM or `tuning` for baseline plus Optuna search.
- `tuning.n_trials` / `timeout_seconds`: hard search budgets; the first limit reached stops the search.
- `tuning.cv_folds`: fixed stratified folds created from Train only.
- `tuning.min_bad_samples_per_fold`: reject tuning when a validation fold lacks statistical support.
- `tuning.objective_metric`: `ks` (default) or `auc`; selects the validation metric used by Optuna.
- `tuning.auc_gap_penalty`: penalize the selected metric's mean Train-minus-validation overfitting gap.
- `tuning.fold_std_penalty`: penalize instability of the selected metric across folds.
- `tuning.search_space.*`: bounded LightGBM ranges for TPE proposals.
- `model.early_stopping_metric`: `ks` (default) or `auc`; the custom evaluation callback records both metrics and uses this one for Test early stopping.
- `model.metric`: retained for configuration compatibility and reporting; KS is calculated outside LightGBM's native metrics.
- `model.*`: LightGBM binary objective, sampling, regularization, iteration, and early-stopping settings.

### Metric convention

IV, KS, and PSI use the Toad backend by default (`toad>=0.1.7,<0.2`). EDA
calculates IV on the fitted reporting bins, model evaluation calculates KS on
the predicted bad-probability score, and score/monthly stability PSI uses the
Train/earliest-month bin support. The selected backend is recorded in EDA's
`metrics_backend` column and model metric tables. A missing Toad installation
falls back to the previous implementation with a warning; install the pinned
dependency in `assets/requirements.txt` to make the production Skill run
strictly Toad-backed.

The per-bin `binning_detail.ks` field remains a cumulative distribution detail
for charting; the headline feature KS in `univariate_overview.ks` is the Toad
two-sample statistic.

Type decisions, category dictionaries, rare-category grouping, IV, monthly stability, and redundancy selection are fitted on Train only. Test and OOT receive the frozen transform; unseen categories do not modify it. PSI thresholds are policy starting points, not universal statistical cutoffs, so the default stability action is review rather than automatic deletion.

The default tuning objective is mean fold validation KS minus configured overfitting and stability penalties. Set `tuning.objective_metric: auc` when AUC is the business objective. The training callback records both AUC and KS; `model.early_stopping_metric` determines which Test curve controls early stopping. The pipeline always retains a configured baseline for comparison. The initial defaults and search ranges are starting values, not a production tuning policy. Never optimize with OOT results.

## Output interpretation

- Compare Train and Test AUC/KS to detect overfitting.
- Read the AUC and KS training curves around `best_iteration`; a widening Train/Test gap indicates overfitting.
- Treat OOT AUC/KS as unreliable when bad-event counts are very small or a period has one target class.
- Use score PSI to detect population shift, not to prove model quality.
- Review feature meaning and leakage risk even when IV and importance are high.
