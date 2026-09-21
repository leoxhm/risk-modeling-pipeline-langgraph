# Model configuration node

`python -m workflow model-config` is the automatic composite boundary for
modeling. It requires completed `data-read` and `feature-processing` inputs,
then creates or reads `configs/node_configs/model-config.yaml` with:

- sample treatment and class-imbalance policy;
- Train/Test/OOT split settings;
- feature preprocessing and selection thresholds;
- LightGBM baseline parameters;
- Optuna sampler, objective, budget and bounded search space.
- Train-only rolling validation (`cv_strategy`, `cv_folds`, `validation_months`,
  `gap_months`, `min_train_months`); every trial reuses the same chronological
  windows and Test/OOT are held out.

The first invocation validates the YAML, writes one compact decision table and
pauses with `waiting_confirmation`; it does not train before that confirmation.
After the user confirms the table, `workflow confirm model-config` invokes the
`training-tuning` executor in-process. That executor also performs model review,
report generation and model export; there is no separate report-delivery
workflow node. Training outputs are written under `outputs/training-tuning/`.

The chat response must contain only one four-column decision table (split/reason,
tuning method and model choice). Do not repeat generic “核心信息/处理方式”
tables or render the same table again from the JSON payload; detailed metrics
remain in the offline HTML/XLSX reports. Approval files are kept outside
`node_configs` so editable YAML and execution evidence do not get mixed.
