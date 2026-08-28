# Sample diagnosis node

This is the third user-facing stage, after the confirmed `data-read` and
`eda-analysis` nodes. It is deliberately independent from EDA and reloads the
source data in a fresh process.

The first invocation reads `configs/node_configs/sample-diagnosis.yaml`,
calculates class balance, monthly continuity/bad-rate changes, duplicate IDs,
missing/invalid labels, row missingness and latest-month completeness, then
writes `sample_diagnostics.json` and waits for confirmation. It does not split
data, apply `class_weight` or train a model.

After the user confirms the single `sample-diagnosis.yaml`, rerun with
`--confirm-config`. The node writes `sample_treatment_policy.json` and records
the configured duplicate, missing-target, all-null-feature, high-missing-row
and incomplete-month actions in the manifest and approval artifact. It never
modifies or saves treated data; later feature-processing/modeling nodes apply
the policy.
The selected class-imbalance action is recorded as a hint for the later
training node rather than applied here.
