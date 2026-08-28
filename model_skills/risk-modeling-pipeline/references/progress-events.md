# Modeling progress events

The Python engine is the source of truth for workflow status. An assistant may summarize an event, but must not fabricate node completion.

## User-facing node order

The UI exposes eight stable stages. Confirmation is a state on the relevant
stage, not an extra node:

1. `data-read`
2. `eda-analysis`（UI 显示为“EDA分析”）
3. `sample-diagnosis`（UI 显示为“样本诊断”）
4. `feature-processing`
5. `model-config`
6. `training-tuning`
7. `model-review`
8. `report-delivery`

The Python engine may still emit detailed internal IDs such as `profiler`,
`feature-selection`, or `model-report`. Consumers should map those IDs to the
eight stages above; the detailed events remain in `run_events.jsonl` for audit
and debugging. A baseline-only configuration emits `tuning=skipped`, which is
shown as progress inside `training-tuning`.

For OpenCode node execution, invoke `python -m workflow <node-id>` and give
each invocation its own node output directory (for example
`outputs/run-001/data-read` and `outputs/run-001/sample-diagnosis`). The node
reloads the source and emits its own `run_events.jsonl`/`run_state.json`; do not
assume a previous node's Python process or in-memory table still exists.

## Statuses

- `pending`: known but not started.
- `running`: deterministic execution started.
- `waiting_confirmation`: execution is paused for explicit user confirmation.
- `success`: the operation returned successfully and declared artifacts were written.
- `failed`: the operation raised an error.
- `skipped`: the node does not apply to this run.

## Outputs

Each executable run directory contains three top-level folders:

- `models/`: native LightGBM, pickle/bundle, and optional PMML files.
- `reports/`: `data_eda_report.xlsx` and `model_report.xlsx` when selected.
- `artifacts/`: cleaned data, metric/detail tables, configs, review evidence,
  `run_events.jsonl`, and `run_state.json`.

The preparation directory remains a confirmation packet and can contain its
JSON/YAML evidence directly. For executable runs, the progress files live in
`artifacts/`:

- `run_events.jsonl`: append-only events for audit and streaming consumers.
- `run_state.json`: latest status per node for reconnect and refresh recovery.

The CLI also prints an invisible marker that OpenCode can parse from tool output:

```text
<!-- opencode-modeling-progress:{"schema_version":1,"event_id":"...","run_id":"run-001","node_id":"profiler","main_node_id":"sample-diagnosis","status":"success","timestamp":"2026-08-24T08:00:00+00:00","summary":"字段画像已生成","artifacts":["/absolute/path/column_profile.csv"],"duration_ms":218} -->
```

The invisible marker and the JSONL row use the same event object. It includes
`schema_version`, `event_id`, `run_id`, the detailed `node_id`, the grouped
`main_node_id`, `timestamp`, `summary`, absolute artifact paths, and
`duration_ms`. OpenCode stores the terminal event on the matching main workflow
node and shows its summary, elapsed time, run ID, and produced artifact names.
Use the same `--run-id` across the preparation, approval, and execution commands
when those commands belong to one user-visible run.

The `sample-diagnosis`, `feature-processing`, `model-config`, and
`training-tuning` stages may become `waiting_confirmation` after their evidence
or result artifacts are available. The stage becomes `success` only after its
declared work and artifacts are complete. `report-delivery` becomes `success`
only after the workbook subprocesses and model-file writes exit successfully.
