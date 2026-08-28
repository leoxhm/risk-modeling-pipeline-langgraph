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

For configuration-gated nodes, the first invocation must create/copy the
workspace YAML files and return `awaiting_user_confirmation`. Only a later
invocation with `--confirm-config` may perform the confirmed transformation.

`common.py` is intentionally limited to path resolution, YAML/config loading,
Polars reload and JSON writing. Node-specific decisions belong in the node
module so that a later node can have a different workflow while sharing the
same input contract.

Current implementations:

- `data_read/run.py`: bootstrap read, role suggestions, contract validation and
  `data_read_summary.json`.
- `eda_analysis/run.py`: independent reload of the confirmed data-read
  configuration, EDA YAML confirmation, EDA analysis and report generation.
- `sample_diagnosis/run.py`: standalone sample diagnostics and user-confirmed
  sample treatment. It does not run EDA, split data, apply class weights or
  train a model; the batch runner may still use its historical compatibility
  path when explicitly requested.

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
└── report_delivery/
```

Every public node has a dedicated directory, even when its implementation is
still a placeholder. This keeps node-specific YAML, workflow code, tests and
future assets co-located as the remaining modeling stages are enabled.

The remaining public node IDs should be added as modules using the same
contract before OpenCode enables them for production execution.
