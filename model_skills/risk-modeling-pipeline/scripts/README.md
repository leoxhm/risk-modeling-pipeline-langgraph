# Python engine boundaries

The Python engine is organized by responsibility under `scripts/`. The Skill
and OpenCode UI should call a public `workflow` node entry point, not reach
into individual domain modules. Every node is restartable and owns its own
read/config/validate/output sequence.

```text
workflow/       User-facing stages, dispatch, and standalone node workflows
  nodes/        One executable module per node; shared context only
data/           Polars loader, schema contract, and deterministic profiler
preprocessing/  Row cleaning, sample diagnostics, and Train-fitted transforms
eda/            IV/WOE/PSI/correlation analytics and EDA workbook payloads
modeling/       Split, feature selection, LightGBM, Optuna, evaluation, reports
metrics/        Toad-backed IV/KS/PSI adapter with legacy compatibility
reporting/      XlsxWriter-based portable EDA and model report exporters
ai/             Advisory packets, experiment planning, approval, model review
progress.py     Append-only event stream and reconnectable run state
logger/         Shared structured logging
```

## Dependency direction

```text
workflow / CLI
    ├── ai
    ├── preprocessing ── data
    ├── eda ──────────── data
    └── modeling ─────── preprocessing, eda, ai, data
```

- `data`, `preprocessing`, `eda`, and `modeling` perform deterministic work.
- `ai` formats evidence into proposals and reviews; it does not calculate
  metrics or silently mutate a contract.
- `metrics` is the only boundary for IV, KS, and PSI conventions. The default
  backend is Toad; if an old environment has not installed Toad, it emits a
  warning and uses the previous deterministic implementation.
- `workflow` names the stages exposed to users and maps detailed engine events
  to those stages. `python -m workflow <node-id>` dispatches a standalone node;
  omitting `<node-id>` keeps the compatibility batch runner.
- `progress.py` is the integration boundary for OpenCode, API callers, and
  reconnectable UIs.

The eight stages are intentionally coarser than Python modules. A refactor of
an internal module must not change the stage IDs or confirmation semantics.
When adding a node, copy the `nodes/` contract: parse project/config paths,
load the relevant YAML, call `NodeContext.reload_data()`, validate inputs,
write node-scoped artifacts, and emit progress events.
