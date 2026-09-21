# Project architecture

The Skill and its deterministic Python engine now live under one directory.
`workflow` is the only execution entry point; the former top-level `src/` tree
and `run_pipeline.py` entry point are no longer part of the layout.

```text
model_skills/
└── risk-modeling-pipeline/
    ├── SKILL.md                         # conversational Skill contract
    ├── agents/openai.yaml               # display metadata
    ├── assets/                          # dependency and config templates
    ├── references/                      # architecture, gates and operations
    └── scripts/                         # merged Python engine
        ├── workflow/
        │   ├── __main__.py              # dispatches one node or batch mode
        │   ├── nodes/                   # independently executable node workflows
        │   │   ├── README.md             # standalone node contract
        │   │   ├── common.py            # shared context, paths and reload helper
        │   │   ├── data_read/           # data-read node folder
        │   │   │   └── run.py
        │   │   ├── eda_analysis/        # dedicated EDA node folder
        │   │   │   └── run.py
        │   │   ├── sample_diagnosis/    # standalone sample diagnosis node
        │   │   │   └── run.py
        │   │   ├── feature_processing/  # feature processing node
        │   │   │   └── run.py
        │   │   ├── model_config/        # model configuration node
        │   │   ├── training_tuning/     # training/tuning node
        │   │   ├── model_review/        # model review logic
        │   │   └── report_delivery/     # legacy compatibility placeholder
        │   ├── runner.py                # compatibility batch orchestration
        │   ├── definition.py            # public node IDs and event mapping
        │   └── node_config.py           # per-node YAML loading/validation
        ├── data/                         # Polars loader, contract, profiler
        ├── preprocessing/                # sample and feature preparation
        ├── eda/                          # metrics and EDA report
        ├── modeling/                     # split, LightGBM, Optuna and report
        ├── metrics/                      # Toad IV/KS/PSI adapter
        ├── ai/                           # advisory and approval artifacts
        ├── reporting/                    # workbook/report presentation helpers
        ├── logger/                       # structured logging helpers
        └── progress.py                   # run events and reconnectable state

user-workspace/
├── data.csv
├── configs/
│   ├── data_contract.yaml
│   ├── node_configs/
│       ├── data-read.yaml
│       ├── eda-analysis.yaml
│       ├── sample-diagnosis.yaml
│       └── feature-processing.yaml
│   └── approvals/                    # hash-bound user confirmations
│       ├── data-read.approval.json
│       └── ...
└── outputs/
    └── run_001/
        ├── models/
        ├── reports/
        └── artifacts/
```

## Workflow entry point

Each public node is implemented as a standalone module or folder under
`workflow/nodes/`. OpenCode can execute only the selected node, and every node
creates its own context, loads its own YAML, and reloads the source data. The
shared helper contains only path/config/loader plumbing; the node module owns
its workflow and output contract:

Workspace bootstrap copies only the selected node's template and required
prerequisite templates. `data_contract.yaml` is the deliberate exception: it
is a shared, user-confirmed contract consumed by every node, while node YAMLs
hold stage-specific parameters and overrides. Approval manifests are written
to the sibling `configs/approvals/` directory; the engine only falls back to
the old `configs/node_configs/*.approval.json` location when reading legacy
workspaces.

```text
workflow.__main__
    ├── data-read       → workflow.nodes.data_read.run
    │                       ├── data-read.yaml (read options + role suggestions)
    │                       ├── data.loader.load_table
    │                       └── data.contract.validate_contract
    ├── eda-analysis     → workflow.nodes.eda_analysis.run
    │                       ├── data-read.yaml (reload source consistently)
    │                       ├── eda-analysis.yaml (EDA policy)
    │                       └── EDA report + diagnostics
    ├── sample-diagnosis  → workflow.nodes.sample_diagnosis.run
    │                       ├── data-read.yaml (reload confirmed roles)
    │                       ├── sample_diagnosis.template.yaml (Skill defaults)
    │                       └── diagnostics + summary artifacts (no confirmation gate)
    ├── feature-processing → workflow.nodes.feature_processing.run
    │                       ├── feature-processing.yaml (selection + type policies)
    │                       └── statistics + processed data after confirmation
    └── no node ID         → workflow.runner (batch/compatibility mode)

`model-config` is the final modeling composite: after confirmation it runs
training/tuning, model review, report generation and model export. The former
`report-delivery` node is not exposed as a selectable workflow stage.
```

Run one node from the repository root with:

```bash
PYTHONPATH=risk-modeling-pipeline/scripts \
python -m workflow data-read \
  --project-root . \
  --engine-root . \
  --data data.csv \
  --data-contract configs/data_contract.yaml \
  --node-config-dir configs/node_configs \
  --output-dir outputs/data-read

PYTHONPATH=risk-modeling-pipeline/scripts \
python -m workflow eda-analysis \
  --project-root . \
  --engine-root . \
  --data data.csv \
  --data-contract configs/data_contract.yaml \
  --node-config-dir configs/node_configs \
  --output-dir outputs/eda-analysis
```

The node commands accept `--engine-root` for a stable OpenCode command
contract, but imports are resolved from the configured `PYTHONPATH`. The
legacy batch runner is still available without a node ID for `check`,
`prepare`, `approve`, `eda`, `model` and `all`; it is not the preferred UI
integration path for node-by-node execution.

The first invocation of `data-read` bootstraps missing files from `assets/` and
returns `waiting_confirmation` for field-role confirmation. EDA, sample
diagnosis, feature processing and model-config use their current YAML/defaults
and return result summaries with a result-confirmation gate; sample diagnosis
uses defaults and does not expose a YAML form. `--confirm-config`
remains available for backwards-compatible advanced runs; the user-facing
workflow uses `--confirm-config` for `data-read` and `--confirm-result` for
the later nodes.

Data source resolution is deterministic: `--data` takes precedence, then
`data.input_path` in the workspace contract (relative to that contract), and
finally a unique supported file directly under `--project-root`. Zero or
multiple discovered files are an explicit error requiring user selection.

## Stable boundaries

- `workflow` owns public stage IDs, node configuration and node dispatch.
- `workflow/nodes/<node>/run.py` (or its compatibility module) owns one
  independently executable workflow. A
  node may reload data and emit artifacts without relying on Python state from
  a previous node process.
- `data` validates the input contract before transformations or modeling.
- `preprocessing` handles row/sample policy and Train-fitted transforms.
- `eda` computes deterministic evidence; `modeling` consumes it for training.
- `ai` formats evidence and approval artifacts; it cannot silently approve.
- `progress.py` is the status interface for OpenCode and API consumers.

The eight public node IDs remain stable. The first three executable node modules
are `data-read`, `eda-analysis` and `sample-diagnosis`; the UI displays them as
“读取数据与契约校验”、“EDA分析” and “样本诊断”. The remaining nodes keep their YAML
templates and can follow the same module contract. Internal Python modules are
co-located under `scripts/`, so consumers do not need a separate `src/` path.
The Skill remains one cohesive Skill, while OpenCode selects a concrete node
subcommand for each user-confirmed step. The batch `--steps` interface remains
for compatibility and offline full-run execution.

## EDA report reuse

The workbook keeps the useful content of `数据源分析EDA/eda/` while moving the
calculation boundary to the current Polars engine:

| Reference content | Current output |
|---|---|
| Sample/monthly label distribution | `3.月度样本` |
| Variable overview and quality profile | `2.数据质量` + `4.字段与单变量分析` |
| Chi-square/WOE/IV/KS binning detail | `7.分箱明细` |
| Toad PSI against a fixed base month | `5.稳定性分析` |
| Monthly KS/AUC/PSI/Lift | `8.月度区分度` |
| Toad-compatible KS buckets and Lift | `9.KS十分位` |
| Correlation analysis | `6.相关性分析` |
| Reference charts | `reports/data_eda_report.html` offline dashboard plus `1.EDA总览` Excel dashboard |

The old Pandas/OpenPyXL exporter is not imported directly: metric calculation
is centralized in `eda/analytics.py`, Toad calls remain behind
`metrics/toad_adapter.py`, and the presentation layer uses XlsxWriter so the
same report can be generated in Linux/OpenCode without a Node runtime.
