---
name: risk-modeling-pipeline
description: "Use ONLY when the user explicitly asks to start, run, continue, or confirm an automatic risk-modeling task (for example: “开始建模”, “运行自动建模”, “确认进入样本诊断”). Do NOT use for greetings, general questions, provider/configuration help, or merely opening AI 建模模式 or selecting workflow nodes. When there is no explicit modeling intent, respond normally and wait for the user to start the task. Once explicitly started, run the human-gated structured-data risk modeling pipeline with Polars, LightGBM, AI-assisted diagnostics, and Optuna tuning."
---

# Risk Modeling Pipeline

## Activation gate

Before loading this Skill, check the latest user message. A greeting or a
general question is not permission to inspect files, call the workflow entry
point,
or change workflow state. Only an explicit request to start/continue modeling
or an explicit confirmation at a modeling gate activates the workflow.

Run the built-in repository engine through the installed Skill's
`risk-modeling-pipeline/scripts/workflow` package. The
Skill installation owns the engine code and Python runtime; the user's current
project is only a modeling workspace containing data and confirmed YAML files.
The user-facing
workflow has eight stages; the engine may expose more detailed sub-events, but
those are implementation details and must not become extra user decisions.
Keep schema decisions, sample treatments, feature policy, model parameters and
model acceptance human-confirmed; keep profiling, splitting, filtering,
tuning, training, scoring and metrics deterministic.

The Skill remains one cohesive Skill; the eight UI nodes are an execution plan,
not eight separately installed Skills. For OpenCode's node-by-node execution,
invoke the selected node subcommand directly, for example
`python -m workflow data-read ...` or
`python -m workflow eda-analysis ...`, `python -m workflow sample-diagnosis ...`
or `python -m workflow feature-processing ...`.
Each node is a fresh process: it
loads the shared `data-read.yaml` reading parameters, reloads the source data,
loads its own node YAML, validates its inputs, and writes node-scoped outputs.
This avoids relying on Python memory left by an earlier node. The shared
`workflow.runner` remains available for `check`/`prepare`/`approve` and full
batch execution; its `--steps` option is a compatibility path, not the
preferred UI integration path.

When OpenCode supplies a selected-node list, it is a strict whitelist. Execute
only those IDs in their listed order and stop after the last selected node is
complete or waiting for confirmation. Do not ask whether to enter, or invoke,
an unselected next node. A later stage requires a new user-created modeling
task with that node selected.

## Eight-stage workflow

1. **读取数据与契约校验** — load CSV/Excel/Parquet with Polars and validate
   the configured ID, date, target and labels.
2. **EDA分析** — immediately after data-read, report missingness,
   duplicate IDs, target completeness, class balance, monthly volume/bad-rate
   changes, IV/KS/PSI and correlations. Ask the user to confirm the EDA node
   YAML (including binning and metric thresholds) before continuing.
3. **样本诊断** — after EDA, analyze class imbalance, missing months, monthly
   bad-rate shifts, duplicate IDs, missing labels and incomplete latest months.
   Ask the user to confirm the diagnostics and treatment actions in
   `configs/node_configs/sample-diagnosis.yaml`. This stage only records the
   policy; later preprocessing applies it. It does not split data or apply
   class weights.
4. **特征预处理与筛选** — fit type handling on Train only, encode or drop
   string-like fields according to policy, then apply missing-rate, IV,
   correlation and PSI/stability rules. The first pass writes the evidence
   tables plus a self-contained `feature_processing_report.html` with metric
   charts; ask the user to confirm the YAML thresholds and preprocessing plan.
5. **生成模型配置** — propose the split, OOT, LightGBM, class-imbalance and
   Optuna settings. Ask the user to confirm the final YAML before training.
6. **模型训练与调参** — always keep a baseline, optionally run bounded
   Optuna Train-only CV, compare on Test, and validate on OOT last. Report the
   result and wait for acceptance or a request to revise parameters.
7. **模型审查** — run deterministic checks and the AI red-team review for
   overfit, underfit, Test→OOT degradation, score drift, weak Lift ordering,
   monthly instability, early stopping and concentrated feature importance.
   The review writes evidence-backed diagnosis and prioritized remediation
   suggestions; it never makes an automatic production approval.
8. **报告与模型交付** — write EDA/model workbooks, model files, scores,
   configs, evidence and progress state. Never label the model production-ready
   automatically.

The confirmation gates are represented by `waiting_confirmation` after the
selected node configuration and evidence are ready. In v1 this applies to
`data-read`, `eda-analysis`, `sample-diagnosis` and `feature-processing`; later modeling stages use
the same pattern.
Do not create a separate “approval” node in a UI or claim that a stage is
complete from a chat response alone.

## Mandatory workflow

1. Locate the current user workspace containing the input data, `configs/` and
   any user-provided output directory. Do not require the workspace to contain
   `scripts/` or the Skill engine source; those are owned by this built-in
   Skill.
2. Resolve the Skill engine root from the installed Skill package. Inspect the
   input and copy only the templates required by the selected node and its
   prerequisites from the Skill's `assets/` when configurations do not exist.
   The shared `data_contract.yaml` is always available because it is the
   cross-node data contract. The standalone nodes perform this bootstrap themselves; the
   source templates remain in the Skill package and are never read from the
   engine's old test workspace.
Use the engine root's `.venv/bin/python` when that virtual environment exists;
the virtual environment is never under `risk-modeling-pipeline/scripts/`.
   do not silently fall back to the JupyterLab or system Python interpreter.
   Pass the user workspace as `--project-root`. Resolve the source data in this
   order: explicit `--data`, `data.input_path` in the workspace contract, then
   automatic discovery of exactly one root-level `.csv`, `.parquet`, `.xlsx` or
   `.xls` file. If discovery finds zero or multiple candidates, stop and ask
   the user to select the source file; never guess between datasets.
3. For an OpenCode UI modeling task, execute the selected standalone node
   directly. Do **not** run a separate `--mode check`, `--mode prepare`,
   `--mode all`, `print('OK')`, `ls`, `find`, or other preflight command: the
   node itself validates its runtime and inputs. For the v1 data path, call
   `python -m workflow data-read ...` first without `--confirm-config`; show
   the generated `data-read.yaml` and wait for the user's confirmation. After
   confirmation, rerun only `data-read` with `--confirm-config`. The legacy
   batch runner may use `--mode check` or `--mode prepare` only when the user
   explicitly asks for an offline/compatibility run.
4. Only after the confirmed data-read succeeds, call the selected EDA node
   (`python -m workflow eda-analysis ...`). Its first call creates
   `eda-analysis.yaml` and waits; its confirmed call performs EDA and writes
   the report. EDA does not split
   Train/Test/OOT, apply class weights, or train a model.
5. If `sample-diagnosis` is selected, execute it only after the confirmed EDA
   node. Its first call loads `sample-diagnosis.yaml`, writes sample
   diagnostics and waits; its confirmed call writes the user-approved policy
   artifact only. It never modifies or saves treated data, splits data,
   applies class weights or trains a model; `class_imbalance_action` is carried
   forward as a training hint.
6. If `feature-processing` is selected, execute it after the confirmed EDA
   node (and confirmed sample policy when that node was selected). Its first
   call calculates missing-rate/IV/KS/correlation/stability evidence and
   creates `feature-processing.yaml` and the offline
   `feature_processing_report.html` chart page; its confirmed call applies
   the selected type-specific preprocessing and writes the processed dataset.
   It does not split data or train a model.
7. Read the artifacts for the current node and ask only the decisions relevant
   to that node. Field roles and read options belong to `data-read.yaml`;
   EDA thresholds, bins, PSI baseline and report options belong to
   `eda-analysis.yaml`; sample thresholds and treatment actions belong to
   `sample-diagnosis.yaml`. Do not ask for future model split or training choices
   during these three nodes.
8. For later modeling nodes, repeat the same per-node confirmation pattern for
   feature processing, model configuration, training, review and delivery.
   The legacy `prepare`/`approve`/`eda`/`model`/`all` batch commands remain
   available only when the user explicitly requests compatibility or offline
   batch execution.

Never infer target semantics, post-outcome leakage exclusions, blocker acknowledgement, or production approval silently. Never create an approval on the user's behalf. If the data, contract, or node configuration changes after confirmation, rerun the corresponding standalone node and obtain a new confirmation. For the legacy batch interface, rerun `prepare` and obtain a new approval. Read [references/ai-workflow.md](references/ai-workflow.md) for the exact gate and [references/configuration.md](references/configuration.md) when reviewing YAML.

Read [references/architecture.md](references/architecture.md) when changing
the repository layout or integrating a new UI/API consumer.
Read [references/node-configs.md](references/node-configs.md) when editing or
reviewing per-node YAML parameters.

Use the same `--run-id` for related `prepare`, `approve`, and execution commands when a caller supplies one. The engine writes authoritative node events to `run_events.jsonl`, a reconnectable snapshot to `run_state.json`, and machine-readable markers to stdout. Do not manually claim that a node completed. Read [references/progress-events.md](references/progress-events.md) when integrating a UI, API, agent runner, or monitoring process.

## Check the environment

The Skill uses a dedicated Python runtime owned by the installed engine. The
expected layout is `<engine-root>/.venv/bin/python`, created from
`assets/requirements.txt` with Python 3.11. Do not use the user's active
Jupyter, system, or OpenCode JavaScript runtime as a silent fallback. If the
dedicated interpreter is missing or the check reports `ready: false`, stop and
ask for environment installation before reading data or running a workflow.

Run the following only when the user explicitly asks for an environment
diagnostic (it is not a required step before a UI node):

```bash
PYTHONPATH=/path/to/builtin/model_skills/risk-modeling-pipeline/scripts \
  /path/to/builtin/model_skills/.venv/bin/python -m workflow \
  --mode check \
  --engine-root /path/to/builtin/model_skills
```

Require Python packages listed in `assets/requirements.txt`. Excel reports are generated with the portable Python `XlsxWriter` backend and do not require Node.js or `@oai/artifact-tool`. The legacy `--node-executable` option remains accepted for command compatibility but is not used by the current report exporter.

Each selected node has an editable file under
`configs/node_configs/<node-id>.yaml`. The executor validates its `node_id`,
version and parameter schema before reading or transforming data, and binds the
file path and SHA-256 into the confirmation/approval artifacts. In v1 the
supported first phase is `data-read`, followed by `eda-analysis` (displayed as
“EDA分析”) and, when selected, the independent
`sample-diagnosis` node (displayed as “样本诊断”), and then
`feature-processing`; model configuration, training, review and delivery node
configs are templates for the next phase and are not run unless explicitly
selected.

For `data-read`, the executor loads the YAML with
`workflow.node_config.ensure_node_configs` before opening the source file. The
optional `parameters.id_col_nm`, `dt_col_nm` and `label_col_nm` fields define
the effective ID, observation-date and target columns. A null value falls back
to the corresponding role in `data_contract.yaml`; a supplied value is
validated and then passed into contract validation, sample treatment, EDA and
modeling. File encoding, Excel sheet and schema inference options are passed
directly to the Polars loader.

The standalone `data-read` node performs a bootstrap read, then fills only
missing role fields in the template using the contract and deterministic
profiler rules. The host AI may edit the same YAML with a reviewed suggestion.
The first invocation returns `waiting_confirmation` and writes a confirmation
summary; it must not execute the confirmed read path yet. If the user edits or
confirms the YAML, rerun the node with `--confirm-config` so the evidence
matches the approved roles.
After explicit user approval, the selected standalone node reloads that YAML
and reads the source again; the approved YAML hash prevents an unreviewed
change from affecting execution. The `prepare`/`eda`/`model`/`all` commands
preserve the same behavior for compatibility batch runs.

The confirmed data-read node also writes
`configs/node_configs/data-read.approval.json`. The standalone EDA node checks
this marker and its hash before reloading the source, so changing
`data-read.yaml` forces a new data-read confirmation before EDA can start.

## Prepare and confirm

Generate diagnostics without training:

```bash
PYTHONPATH=/path/to/builtin/model_skills/risk-modeling-pipeline/scripts \
/path/to/builtin/model_skills/.venv/bin/python -m workflow \
  --mode prepare \
  --planned-mode all \
  --project-root /path/to/user/workspace \
  --engine-root /path/to/builtin/model_skills \
  --data /path/to/user/workspace/data.csv \
  --data-contract /path/to/user/workspace/configs/data_contract.yaml \
  --model-config /path/to/user/workspace/configs/model_config.yaml \
  --sample-config /path/to/user/workspace/configs/sample_config.yaml \
  --output-dir /path/to/user/workspace/outputs/prepare_001
```

After explicit confirmation, create the hash-bound approval:

```bash
PYTHONPATH=/path/to/builtin/model_skills/risk-modeling-pipeline/scripts \
/path/to/builtin/model_skills/.venv/bin/python -m workflow \
  --mode approve \
  --project-root /path/to/user/workspace \
  --engine-root /path/to/builtin/model_skills \
  --confirmation-request /path/to/outputs/prepare_001/confirmation_request.json \
  --confirmed-by "confirmed-user" \
  --confirm-node data-read \
  --confirm-node sample-diagnosis \
  --approval-file /path/to/outputs/prepare_001/approval_manifest.json
```

Add `--acknowledge-finding CODE` once per confirmed blocker when required.

## Execute the approved run

Execution modes are:

- `eda`: clean confirmed data and create `reports/data_eda_report.xlsx` plus detailed CSV/Parquet outputs under `artifacts/eda`.
- `model`: perform row-level preparation; split data; fit feature typing, categorical mappings, and feature selection on Train only; apply the frozen transformations to Test/OOT; always fit a baseline; optionally tune LightGBM with Optuna on Train-only folds; compare the final candidate once on Test; validate OOT only after selection; write model files under `models/`, and create `reports/model_report.xlsx`.
- `all`: run EDA followed by modeling in one run directory.

Example:

```bash
PYTHONPATH=/path/to/builtin/model_skills/risk-modeling-pipeline/scripts \
/path/to/builtin/model_skills/.venv/bin/python -m workflow \
  --mode all \
  --steps data-read,sample-diagnosis,feature-processing,model-config,training-tuning,model-review,report-delivery \
  --project-root /path/to/user/workspace \
  --engine-root /path/to/builtin/model_skills \
  --data /path/to/user/workspace/data.csv \
  --data-contract /path/to/user/workspace/configs/data_contract.yaml \
  --model-config /path/to/user/workspace/configs/model_config.yaml \
  --sample-config /path/to/user/workspace/configs/sample_config.yaml \
  --approval-file /path/to/user/workspace/outputs/prepare_001/approval_manifest.json \
  --output-dir /path/to/user/workspace/outputs/run_001
```

Do not use Test or OOT to fit category dictionaries, rare-category grouping, type-based feature decisions, quality thresholds, IV, stability, or redundancy selection. Do not use OOT for early stopping, threshold selection, candidate selection, or parameter tuning. In tuning mode, Optuna uses fixed stratified Train-only folds and a penalized validation-AUC objective. Test is used only for final baseline-versus-tuned comparison and early stopping; OOT is the final temporal validation. Treat monthly AUC/KS as unavailable when a month contains only one target class.

## Review outputs

Verify that the run directory contains the applicable deliverables:

- EDA: cleaned data, field profile, quality decisions, binning/WOE, IV/KS/AUC, monthly PSI, monthly KS/AUC/PSI/Lift, Toad KS buckets, correlations, and `reports/data_eda_report.xlsx`. The workbook reuses the reference EDA content in a Linux-compatible XlsxWriter layout: `1.EDA总览`, `2.数据质量`, `3.月度样本`, `4.字段与单变量分析`, `5.稳定性分析`, `6.相关性分析`, `7.分箱明细`, `8.月度区分度`, and `9.KS十分位`. PSI compares each event month's fitted-bin distribution with the earliest available baseline month; numeric-feature KS uses Toad on the raw finite values, while IV and KS bucket detail use Toad on the displayed/fitted bins.
- Sample diagnosis: `sample_diagnostics.json` and
  `sample_diagnosis_confirmation.json` before confirmation; after confirmation,
  `sample_treatment_policy.json`, `sample_diagnosis_manifest.json` and the
  `sample-diagnosis.approval.json` gate. The node records duplicate IDs,
  missing/invalid labels, class imbalance, missing months, monthly bad-rate
  shifts, high-missing rows and incomplete latest-month evidence. It does not
  split data or apply class weights.
- Modeling: copied configuration, split summary, Train-fitted preprocessing plan and model matrix, feature decisions with stability evidence, baseline/tuned comparison, Optuna study database, trial and CV-fold history, best parameters, LightGBM native/pickle model artifacts, optional PMML status, training history, feature importance, scored data, KS/AUC/PSI/Lift tables, monthly performance, AI red-team review (`ai_model_review.json` with diagnosis, evidence, thresholds and recommendations), and `model_report.xlsx` (including `13.诊断建议`).

Model delivery writes `lightgbm_model.txt`, `lightgbm_model.pkl` and
`model_bundle.pkl`. PMML is optional: pass `--pmml-converter` or set
`JPMML_LIGHTGBM_JAR`; if the Java converter is unavailable, keep the native
and pickle files and report the reason in `pmml_export_status.json`.

Flag train/test gaps, low OOT bad-event counts, score drift, and unstable monthly results. Describe the model as a workflow-validation model when statistical support is inadequate. Require explicit user approval before any production-readiness claim.

## Legacy migration

Read [references/legacy-project.md](references/legacy-project.md) only when mapping code from the previous notebook-style implementation.
