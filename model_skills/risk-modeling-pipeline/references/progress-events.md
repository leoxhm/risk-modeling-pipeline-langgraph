# Modeling progress events

The Python engine is the source of truth for workflow status. An assistant may summarize an event, but must not fabricate node completion.

## User-facing node order

The UI exposes six stable stages. Confirmation is a state on the relevant
stage, not an extra node. Model reports and model files are generated inside
the `model-config` composite node:

1. `data-read`
2. `eda-analysis`（UI 显示为“EDA分析”）
3. `sample-diagnosis`（UI 显示为“样本诊断”）
4. `feature-processing`
5. `model-config`
6. `model-review`

The Python engine may still emit detailed internal IDs such as `profiler`,
`feature-selection`, `llm-tuning`, or `model-report`. Consumers should map those IDs to the
six stages above; `model-report` is grouped under `model-config`. The detailed
events remain in `run_events.jsonl` for audit and debugging. A baseline-only
configuration emits `tuning=skipped`, which is shown as progress inside
`model-config`.

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

The append-only `run_events.jsonl` file is the authoritative source for the
training monitor. The OpenCode frontend polls that file through the workspace
file API and merges unseen events into the local session view. Progress markers
remain as a compatibility/notification path, but a truncated chat or shell
preview no longer removes Trial history from the monitor.

The invisible marker and the JSONL row use the same event object. It includes
`schema_version`, `event_id`, `run_id`, the detailed `node_id`, the grouped
`main_node_id`, `timestamp`, `summary`, absolute artifact paths, and
`duration_ms`. During model training, an optional `experiment` object is also
emitted for each completed Optuna Trial or LLM round. It contains the source
(`kind`), sequence number, parameter snapshot, metrics, objective value and
acceptance/best-so-far flags. Optuna search events are intentionally scored with
Train-only rolling CV metrics (`mean_validation_ks`/`mean_validation_auc`) so
the Test set is not used to select a trial. After the best trial is retrained,
one additional `phase: "final_evaluation"` event records its Test AUC/KS and
`improvement_vs_baseline`; this is the row the monitor uses for the Test-side
comparison. OpenCode stores the terminal event on the matching main workflow
node and shows its summary, elapsed time, run ID, produced artifact names and
the retained experiment history.

Training also emits a `phase: "baseline_evaluation"` event before tuning. It
contains the baseline parameters and Train/Test/OOT AUC/KS, so the monitor can
draw real comparison points instead of starting the Test/OOT series with blank
values.

Example experiment event:

```json
{
  "kind": "optuna",
  "trial": 12,
  "total_trials": 50,
  "status": "complete",
  "parameters": {"learning_rate": 0.035, "num_leaves": 31},
  "metrics": {
    "mean_validation_ks": 0.301,
    "objective_metric": "ks"
  },
  "objective": 0.286,
  "best_so_far": true
}
```

The final Optuna evaluation event has the same shape, with
`phase: "final_evaluation"`, `candidate: "optuna_tuned"`, and metrics such as
`test_ks`, `test_auc`, `acceptance_metric`, and `improvement_vs_baseline`. In
the monitor, the project-level “Test / Validation KS” field maps Optuna search
events to `mean_validation_ks` and final-evaluation events to `test_ks`. A
search Trial without the final-evaluation phase should display the validation
value, rather than treating a missing independent Test value as zero.
Use the same `--run-id` across the preparation, approval, and execution commands
when those commands belong to one user-visible run.

The `eda-analysis`, `sample-diagnosis`, `feature-processing` and `model-config`
stages return a result summary after their artifacts are written and wait for
result confirmation. Sample diagnosis uses default parameters and never
modifies data. The `model-config` stage becomes
`success` only after training, model review, workbook generation, and
model-file writes exit successfully. Only `data-read` returns
`waiting_confirmation` for its first field-role configuration pass, while the
later stages use the same status for result confirmation.
