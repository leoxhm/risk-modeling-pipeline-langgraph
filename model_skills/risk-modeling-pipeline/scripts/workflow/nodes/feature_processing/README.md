# Feature processing node

This node has a strict evidence/confirmation boundary:

1. The first invocation reloads data using the confirmed data-read YAML and
   EDA settings, then writes `feature_statistics.csv` (missing rate, IV, KS and
   monthly PSI/stability) plus `feature_correlations.csv`. It creates the
   editable `configs/node_configs/feature-processing.yaml` and waits.
2. The user confirms the selection thresholds and type-specific preprocessing
   methods in that one YAML file.
3. The confirmed invocation applies any confirmed sample policy, selects
   features using the configured missing-rate/IV/correlation/stability rules,
   fits numeric/categorical/text handling, and writes `processed_data.parquet`
   plus auditable selection and preprocessing artifacts. The
   `feature_selection_summary.md/json` files explicitly list retained,
   excluded and review-required fields with reasons.

The node never fits decisions on Test or OOT data. It does not train a model.
