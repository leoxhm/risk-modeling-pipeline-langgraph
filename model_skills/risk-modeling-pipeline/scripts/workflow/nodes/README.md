# Standalone node contract

OpenCode executes a node as an independent Python process. A node must not
depend on an in-memory object produced by a previous node process.

Each node module should follow this boundary:

1. Parse `--project-root`, `--data`, contract/config paths, `--output-dir` and
   `--run-id`.
2. Build `NodeContext.from_args(args, node_id)`. This loads the shared
   `data-read.yaml` and the node's own YAML.
3. Call `context.reload_data()` to read the source with the current read
   parameters. Re-validate the effective contract in the node.
4. Execute only the node's deterministic workflow and emit progress events.
5. Write all artifacts below the node output directory and return a small JSON
   result containing the node ID and artifact paths.

`data-read` confirms the field roles and read configuration. Its first
invocation creates/copies the workspace YAML and returns
`awaiting_user_confirmation`. EDA, sample diagnosis, feature processing and
model-config return `awaiting_user_confirmation` after generating their
evidence; sample diagnosis uses defaults and does not expose a YAML form.
`workflow confirm <node-id>` records result confirmation for the gated nodes
and allows the next stage. The `--confirm-config` flag
remains a backwards-compatible alias for the explicit confirmation pass.
The editable YAML files live under `configs/node_configs/`; hash-bound approval
manifests are written under the sibling `configs/approvals/` directory. For
backward compatibility, an existing approval in `node_configs` may be read but
new approvals are never written there.

`common.py` is intentionally limited to path resolution, YAML/config loading,
Polars reload and JSON writing. Node-specific decisions belong in the node
module so that a later node can have a different workflow while sharing the
same input contract.

Current implementations:

- `data_read/run.py`: bootstrap read, role suggestions, contract validation and
  `data_read_summary.json`.
- `eda_analysis/run.py`: independent reload of the confirmed data-read
  configuration, direct EDA analysis with safe template defaults and report
  generation. EDA does not add a second YAML confirmation gate.
- `sample_diagnosis/run.py`: standalone sample diagnostics and user-confirmed
  sample treatment. It does not run EDA, split data, apply class weights or
  train a model; the batch runner may still use its historical compatibility
  path when explicitly requested.
- `feature_processing/run.py`: feature metrics, self-contained HTML evidence,
  automatic thresholds/type handling and processed-data generation.
- `model_config/run.py`: the automatic pre-training composite. It records
  sample treatment, split, LightGBM and Optuna settings, validates the
  canonical model config, then trains, reviews and exports the model.

Directory convention:

```text
workflow/nodes/
├── data_read/run.py
├── eda_analysis/run.py
├── sample_diagnosis/run.py       # independent sample-diagnosis node
├── feature_processing/           # one folder per future node
├── model_config/
├── training_tuning/
├── model_review/
└── report_delivery/              # legacy compatibility placeholder
```

Every executable node has a dedicated directory. The report-delivery folder is
retained only so older workspaces and imports remain readable; report creation
is now part of the confirmed model-config composite and is not selectable in
the UI.

The remaining public node IDs should be added as modules using the same
contract before OpenCode enables them for production execution.
