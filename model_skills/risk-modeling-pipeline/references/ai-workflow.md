# AI 协作与分阶段确认

助手编排六个用户节点。`data-read` 需要确认字段角色和读取配置；EDA 只生成报告并自动通过；样本诊断、特征处理完成后输出反馈卡片并等待结果确认；模型配置先展示建模决策表并等待确认，确认后才启动训练。用户只有提出修改时才触发当前节点重跑。不得把引擎内部
的 profiler、preflight、cleaning、EDA、tuning 或 report 子事件暴露成新的用户决策。

## Gate sequence

1. After **data-read**, first display the engine-provided Markdown summary
   table (file name, shape, ID/date/Y roles, labels and feature count), then
   validate `configs/node_configs/data-read.yaml` and confirm the read options,
   contract roles and target/date parsing behavior. Never reply with only a
   generic “waiting for configuration” sentence.
2. After the confirmed data-read, execute **EDA analysis** through
   `workflow.nodes.eda_analysis.run`. Read the current
   `configs/node_configs/eda-analysis.yaml` with safe defaults, calculate field
   quality, IV/KS/PSI, monthly discrimination, correlations and the generated
   report. Do not ask the user to fill a second EDA parameter form or confirm
   the report in chat; the node is auto-approved after writing the report. EDA
   must not split Train/Test/OOT, apply class weights or train a model.
3. If selected, execute **sample diagnosis** through
   `workflow.nodes.sample_diagnosis.run` after EDA. Use the Skill's default
   diagnostic parameters to summarize class imbalance, missing months, monthly
   bad-rate shifts, duplicate IDs, missing/invalid labels and incomplete latest
   months. In addition to the raw findings, return one modeling-risk table with
   IV pass ratio, PSI effective/stable ratio, bad-rate range, missingness,
   imbalance and time-integrity checks; every row must explain its threshold.
   Do not wait for YAML confirmation or write a treatment approval;
   findings are advisory warnings only; the node returns the table and waits
   for the user's confirmation before the next selected node starts.
   The node does not modify or save treated data, split data, train a model or
   apply class weights. Treatment choices are confirmed later in model config.
4. After **feature preprocessing and selection**, return one Markdown result table.
   It must show the input/output shape, retained fields, excluded fields (show up
   to eight names with the actual reason and report the total when there are more),
   type-based exclusions, review-required fields, thresholds, IV/KS/PSI ranges,
   and the numeric/categorical/text handling actually applied. Do not include the
   later OOT/Train/Validate time-split strategy in this node's result table; that
   strategy belongs to `model-config`. Do not require a parameter-by-parameter form; apply the
   current YAML immediately. The result is then confirmed before continuing.
   If the user requests a change, edit the YAML and rerun the current node.
5. On the first **model configuration** call, show one concise Markdown decision
   table rather than the full YAML, then pause. The table must contain the
   chronological sample split and its reason, the selected tuning method
   (`none`, `optuna`, or `llm`), and the selected model (currently LightGBM only).
   Do not repeat generic “核心信息/处理方式” tables or render the JSON summary
   again. The three tuning methods are mutually exclusive; never chain Optuna
   and LLM in one run. Only after the user explicitly confirms the table should
   the host call `workflow confirm model-config`; that single command starts
   training, tuning, model review, report generation and model export.
6. The **model-review** view presents the review evidence emitted by the
   modeling composite. It is not followed by a separate report-delivery node.

The confirmation manifest remains the final hash-bound execution gate. If the
host wants to run the stages as separate jobs, each job must use a new request
and approval hash after any contract, sample policy or model configuration
change.

## What AI may do

- Summarize deterministic profiling and data-readiness evidence.
- Propose ID, date, target, feature, and exclusion roles with reasons and uncertainty.
- Organize a baseline-versus-tuning experiment plan within the configured budget.
- Red-team completed model metrics for overfitting, weak OOT support, unstable training, and concentrated feature importance.

AI output is advisory. Python remains the source of truth for counts, hashes, splits, metrics, tuning, and report tables.

For user interaction, keep each node result at a medium level of detail: one
completion sentence, four to six key findings with metric evidence, three to
five small Markdown tables for EDA (overview, IV/KS, PSI, monthly bad rate and
correlation; each table should stay within 5–8 rows), two to three
modeling-oriented suggestions, adjustable decisions and a next-action line. Do not paste
full tables or repeat the HTML/Excel contents; provide the fixed artifact paths
for details. Use “需关注” for advisory findings and never call them blockers
or claim that the data cannot be modeled solely because of an EDA/sample-
diagnosis warning.

逐节点反馈格式：上述摘要之后附一张“节点反馈卡片”，固定包含节点结果、3-5 条
证据、大模型建议、1-3 个可调整的业务决策或 YAML 字段及其影响，以及下一步动作。
同一节点的完整摘要只使用进度事件中的 `summary` 展示一次，不要再把命令 JSON 的同名字段重复渲染。
`data-read` 等待配置确认；其他节点等待结果确认。用户提出修改指令时，先修改当前
节点配置并重跑当前节点。最后一个已选节点反馈确认后停止。大模型建议只解释确定性产物，
不自动改数或替用户做业务确认。

其中 `data-read` 的“节点结果”必须直接使用引擎返回的固定 Markdown 表格，至少包含
数据文件名、文件格式、行列数、ID 字段及唯一率、日期字段及解析率、目标字段（Y）、
好坏标签映射、特征数量/预览、排除字段和读取参数。YAML 和哈希只用于引擎内部校验，
不要在聊天窗口展示。用户可直接确认表格，或用自然语言提出字段/读取策略修改；不要只回复
“等待确认 YAML”，也不要让模型根据日志自行补写这些值。

The per-node templates are maintained in
`references/eda-response-template.md` and
`references/sample-diagnosis-response-template.md` (the index is
`references/response-templates.md`); use them as the response contract rather
than inventing a new layout in each run.

## What AI must not do

- Decide target meaning, label polarity, leakage fields, or observation windows without user confirmation.
- Treat a field-name guess as a confirmed schema decision.
- Acknowledge a blocker or write an approval manifest on the user's behalf.
- Modify data or configurations after confirmation and reuse the old approval.
- Tune, select, or approve a candidate using OOT performance.
- Claim production readiness from an automated report.

## Gate artifacts

The legacy batch `prepare` command creates:

- `column_profile.csv`: deterministic column statistics.
- `sample_diagnostics.json`: class balance, target completeness, duplicate IDs, row missingness, month continuity, monthly target rates, findings, and the proposed treatment actions.
- `ai_preflight.json`: data-readiness findings and evidence.
- `ai_schema_proposal.json`: field-role and leakage-review suggestions.
- `ai_experiment_plan.json`: proposed baseline/tuning sequence and guardrails.
- `feature_proposal.json`, `feature_preprocessing.csv`, and
  `feature_selection.csv`: Train-only type, encoding, quality, IV,
  correlation and stability evidence for the feature result-confirmation gate.
- `model_config_proposal.yaml`: deterministic snapshot of the model parameters
  used before the model result-confirmation gate.
- `confirmation_request.json`: exact roles, blocker codes, file paths, and SHA-256 hashes awaiting confirmation.
- `review_artifacts.node_configs`: the selected node YAML files. The approval
  command requires one `--confirm-node <node-id>` for every selected node.

`approve` creates `approval_manifest.json` only after explicit user confirmation. Every blocker code must be individually acknowledged. Before execution, the CLI rejects a manifest when its mode, path, or file hash does not match the requested run.

## Required confirmation summary

Before asking the user to approve, state:

1. ID fields and their uniqueness meaning.
2. Observation date and available time range. Do not ask for OOT months yet.
3. Target field plus good/bad label mapping.
4. Included features and suspected leakage fields only when present in the
   confirmed read/EDA evidence.
5. EDA-specific quality, binning, PSI baseline and report options are included
   in the generated result overview for reference; do not create a second EDA
   parameter form or ask for class imbalance, Test ratio or training
   parameters during EDA. The result feedback itself still requires confirmation.
6. When sample diagnosis is selected, show its short diagnostic summary and
   default-parameter note. Do not ask the user to confirm
   `sample-diagnosis.yaml`; treatment actions are configured and confirmed in
   the later model-config stage.
7. For later modeling nodes only, ask for OOT/Test, class-imbalance policy,
   Optuna budget and training guardrails.
8. Every blocker with its evidence, consequence, and safer alternative.

If the user changes any item, update the appropriate YAML, rerun `prepare`, and present the refreshed request.
