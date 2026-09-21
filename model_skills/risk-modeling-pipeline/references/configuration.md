# Configuration reference

## Data contract

Use `assets/data_contract.template.yaml` as the starting point. Keep the
result as `configs/data_contract.yaml` in the user workspace; it is the shared
cross-node contract rather than a node-specific parameter file.

Editable node parameters remain in `configs/node_configs/`. Confirmation
manifests are written separately to `configs/approvals/` so users can review
YAML without mixing it with execution evidence. Existing manifests in the old
`node_configs` directory are read only for backward compatibility.

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
- `treatment.duplicate_action`: `keep_first` or `keep_last` (the default is `keep_first`).
- `treatment.missing_target_action`: `drop` (missing labels are excluded before training).
- `treatment.all_null_feature_action`: `drop` (all-null candidate columns are removed).
- `treatment.high_missing_row_action`: `keep` or `drop`.
- `treatment.incomplete_latest_month_action`: `keep` or `exclude` (the default is `exclude`).
- `treatment.class_imbalance_action`: `none` or Train-only `class_weight`.

The former `error` values remain accepted only as legacy aliases so existing
confirmed workspaces do not break: they are treated as `keep_first`, `drop`,
`drop`, and `exclude` respectively. New templates never emit `error`, and all
sample findings remain visible as warnings rather than aborting the run.

Do not use over/under-sampling on Test or OOT. This version deliberately supports class weighting before synthetic sampling because it preserves rows and is easier to audit.

## Model configuration

Use `assets/model_config.template.yaml` as the starting point.

- `split.strategy`: `time` (default) reserves contiguous historical Test months; `random_stratified` is retained only for legacy comparisons.
- `split.oot_months`: reserve the newest distinct months before any Train/Test work.
- `split.test_months`: reserve the months immediately preceding OOT as Test when using the time strategy.
- `split.test_ratio`: stratified fraction taken from the remaining development sample only in the legacy random strategy.
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
- `training.mode`: legacy baseline/tuning compatibility switch. The effective tuning backend is selected by `tuning.method`.
- `training.acceptance_metric`: `ks` (default) or `auc`; compare the tuned candidate with Baseline on the configured Test metric before exporting the final model.
- `training.min_improvement`: minimum non-negative improvement required for the tuned candidate to replace Baseline. If it is not reached, only Baseline is exported as the final model.
- `tuning.n_trials` / `timeout_seconds`: hard search budgets; the first limit reached stops the search.
- `tuning.method`: mutually exclusive tuning backend: `none` (baseline only), `optuna` (Optuna only), or `llm` (LLM proposal/retrain loop only). Templates default to `llm`; the selected method is recorded in `model_summary.json`; Optuna and LLM are never chained in one run.
- `tuning.cv_strategy`: `rolling`（默认，按月份扩展窗口）或 `stratified`（兼容旧版随机折）。
- `tuning.cv_folds`: fixed Train-only validation windows/folds created once and reused by every Optuna trial.
- `tuning.validation_months` / `tuning.gap_months` / `tuning.min_train_months`: rolling CV 每个验证窗口的月份数、训练与验证间隔、首个训练窗口最少月份数。
- `tuning.min_bad_samples_per_fold`: reject tuning when a validation fold lacks statistical support.
- `tuning.objective_metric`: `ks` (default) or `auc`; selects the validation metric used by Optuna.
- `tuning.llm.enabled`: enables the LLM implementation when `tuning.method=llm` (it is ignored for `none` or `optuna`). Each accepted round should improve the frozen Validate metric by `tuning.llm.min_improvement` and stay within the configured search-space bounds. When `guardrails.allow_gap_improvement=true`, a candidate can also be retained when the Train/Validate KS gap shrinks materially while Validate/OOT KS only regresses within the configured small tolerance.
- `tuning.llm.max_rounds`: maximum number of LLM proposal/retrain rounds (default `10`). `base_url`, `model` and `api_key` configure an OpenAI-compatible service directly in the node YAML; when `api_key` is empty, `api_key_env` (default `LLM_TUNING_API_KEY`) is used as a fallback. Missing service credentials safely skip the loop. If `api_key` is written in YAML, treat the workspace as a secret-bearing directory and do not commit or share the file.
- `tuning.llm.run_all_rounds`: defaults to `true` in the templates. The engine consumes the full `max_rounds` budget and records every proposal/retraining result; target, plateau and acceptance thresholds still decide which candidate is retained, but no-improvement rounds do not stop the loop. Provider failures can still stop after the configured consecutive-error threshold.
- `tuning.llm.timeout_seconds` / `request_retries` / `max_consecutive_provider_errors`: 单次请求超时、单轮额外重试次数和连续服务失败终止阈值。临时超时不会直接结束整个 LLM 调参；每个失败轮次会写入历史并继续，只有连续失败达到阈值才停止。
- `tuning.llm.stopping`: early-stop policy. `targets` defaults to Validate KS `0.30`, Validate AUC `0.75`, OOT KS `0.25`, OOT AUC `0.70`; `target_tolerance` defaults to `0.01`; the templates use `plateau_rounds=10` and `plateau_min_improvement=0.001` so small-step tuning is not stopped after two rounds. `guardrails` defaults to Train/Validate KS gap `0.10`, Validate/OOT KS drop `0.08`, OOT degradation versus Baseline `0.03`, and minimum OOT bad samples `30`. With `allow_gap_improvement`, `min_gap_reduction=0.02`, `max_validate_ks_regression=0.005`, and `max_oot_ks_regression=0.01`, a materially smaller Train/Validate gap can be treated as progress. OOT thresholds are informational when the OOT sample has fewer bad cases than the minimum.
- `tuning.llm.verify_ssl` / `ca_bundle`: TLS verification is enabled by default. Set `ca_bundle` to an internal CA PEM when the provider certificate is not in the Python trust store. `verify_ssl: false` is an explicit, insecure last resort for isolated internal testing only.
- `feature-processing.yaml.parameters.llm`: optional EDA-grounded parameter advisor. It receives only aggregate diagnostic/statistical evidence and writes a bounded `*.llm-recommended.yaml` draft; the draft is never applied without editing the original YAML and confirming it. Sample diagnosis uses deterministic Skill defaults and does not call an advisor.
- `llm-tuning/llm_tuning_history.json` / `.csv`: auditable per-round record containing current/proposed parameters, LLM reason, bound validation, candidate score, improvement and acceptance status.
- `tuning.auc_gap_penalty`: penalize the selected metric's mean Train-minus-validation overfitting gap.
- `tuning.fold_std_penalty`: penalize instability of the selected metric across folds.
- `tuning.search_space.*`: bounded LightGBM ranges for TPE proposals.
- `model.early_stopping_metric`: `ks` (default) or `auc`; the custom evaluation callback records both metrics and uses this one for Test early stopping.
- `model.metric`: retained for configuration compatibility and reporting; KS is calculated outside LightGBM's native metrics.
- `model.*`: LightGBM binary objective, sampling, regularization, iteration, and early-stopping settings.

Model evaluation is generated automatically after the final model is selected:
`metrics_by_split.csv` includes AUC, KS, PR-AUC and Brier Score; the report also
contains Train/Test/OOT ROC and Lift/Gain curves, decile bad-rate trends,
calibration bins, deterministic Bootstrap intervals (200 resamples), three
random-seed stability runs, monthly/low-cardinality subgroup metrics, and
permutation feature-importance drift across Train/Test/OOT. These diagnostics
are evidence for review and do not change model acceptance rules.

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

The default tuning objective is mean fold validation KS minus configured overfitting and stability penalties. Each parameter trial uses its own LightGBM early-stopping point in every Train CV fold; `tuning_trials.csv` records the mean/min/max best iteration for comparison. Set `tuning.objective_metric: auc` when AUC is the business objective. The training callback records both AUC and KS; `model.early_stopping_metric` determines which Test curve controls early stopping. The pipeline always retains a configured baseline for comparison. The initial defaults and search ranges are starting values, not a production tuning policy. Never optimize with OOT results.

## Output interpretation

- Compare Train and Test AUC/KS to detect overfitting.
- Read the AUC and KS training curves around `best_iteration`; a widening Train/Test gap indicates overfitting.
- Treat OOT AUC/KS as unreliable when bad-event counts are very small or a period has one target class.
- Use score PSI to detect population shift, not to prove model quality.
- Review feature meaning and leakage risk even when IV and importance are high.
