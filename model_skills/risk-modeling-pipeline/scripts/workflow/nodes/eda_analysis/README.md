# EDA analysis node

This node is the second user-facing stage (the UI label is “EDA分析”).
It runs independently from `data-read` and never reuses an in-memory dataframe.

## Gate

The first invocation:

1. reads the current `data-read.yaml` and source file;
2. checks `data-read.approval.json` and its SHA-256;
3. creates `configs/node_configs/eda-analysis.yaml` when missing;
4. writes `eda_confirmation.json` and returns `awaiting_user_confirmation`.

Only after the user confirms the EDA parameters should the node be rerun with
`--confirm-config`. It then reloads the source, applies the configured cleaning
policy, calculates EDA tables and writes the workbook below `reports/`.

The public command `sample-diagnosis` is a separate node implemented under
`workflow/nodes/sample_diagnosis/`; it runs only after this EDA gate has
completed. Historical batch runs may still retain their combined compatibility
path, but standalone OpenCode execution must use `eda-analysis` for EDA.
