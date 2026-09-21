# Sample diagnosis node

This is the third user-facing stage, after the confirmed `data-read` and
`eda-analysis` nodes. It is deliberately independent from EDA and reloads the
source data in a fresh process.

The node loads the Skill's default `sample_diagnosis.template.yaml`,
calculates class balance, monthly continuity/bad-rate changes, duplicate IDs,
missing/invalid labels, row missingness and latest-month completeness, then
writes `sample_diagnostics.json`, `modeling_risk_summary.json`,
`sample_diagnosis_manifest.json` and a short `node_summary.json`. It also
returns one modeling-risk table covering IV/PSI pass ratios, bad-rate range,
missingness, imbalance and time integrity, with the threshold explained in
every row. All findings are advisory warnings; they do not block a later
selected node. The node does not wait for YAML confirmation, split data,
apply `class_weight` or train a model.

`configs/node_configs/sample-diagnosis.yaml` may still be copied into the
workspace for transparency and backward compatibility, but edits to it are not
used by this diagnostic run. Sample-treatment choices are confirmed later in
the model configuration stage and are not applied here.
