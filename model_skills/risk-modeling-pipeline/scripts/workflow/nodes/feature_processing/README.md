# Feature processing node

This node executes directly with the current YAML/defaults and waits only for
result confirmation. It keeps a readable YAML for audit
and advanced overrides; ordinary users review the generated evidence instead
of filling every parameter:

1. The invocation reloads data using the confirmed data-read YAML and
   EDA settings, then writes `feature_statistics.csv` (missing rate, IV, KS and
   monthly PSI/stability) plus `feature_correlations.csv`. It creates or reads
   the editable `configs/node_configs/feature-processing.yaml`.
2. The invocation applies the current selection thresholds and type-specific
   preprocessing methods, then selects
   features using the configured missing-rate/IV/correlation/stability rules,
   fits numeric/categorical/text handling, and writes `processed_data.parquet`
   plus auditable selection and preprocessing artifacts. The
   `feature_selection_summary.md/json` files explicitly list retained,
   excluded and review-required fields with reasons. The progress summary is a
   single Markdown table: it shows up to eight excluded fields and their actual
   reasons, the total excluded count, thresholds, metric ranges, processing
   methods, and the generated data shape. If the list is longer, the full list
   remains in `feature_selection_summary.md` and the field-level metrics remain
   in `feature_statistics.csv`.

The feature-processing node intentionally does not split samples into
Train/Validate/OOT. It freezes selection and preprocessing evidence on the
current data first; chronological splitting is performed later by `model-config`
so that future observations cannot influence feature decisions.

The node never fits decisions on Test or OOT data. It does not train a model.

The node does not call a separate parameter-advisor service. The host LLM
receives the deterministic evidence from the result and returns a concise
summary plus textual parameter-adjustment suggestions. It never writes or
applies a recommendation YAML automatically.
