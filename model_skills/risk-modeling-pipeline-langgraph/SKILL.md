---
name: risk-modeling-pipeline-langgraph
description: "使用 LangGraph 编排金融风控建模流程，在 data-read、特征处理和模型配置等节点提供可持久化的用户确认与恢复；仅在用户明确启动或继续自动建模时使用。"
---

# LangGraph 风控建模 Skill

本 Skill 是独立实现的 LangGraph 版风控建模试验。它不导入或调用旧版 `risk-modeling-pipeline`；数据读取、契约解析、字段角色推断和后续计算都放在本目录。LangGraph 只负责节点跳转、状态保存和 `interrupt()` 人工确认。

## 触发边界

仅在用户明确要求“开始/继续/确认自动建模”时启动。普通问候、模型咨询、环境配置和仅打开建模面板时不要调用。

## 节点流程

```text
data-read → 用户确认 → eda-analysis（含样本诊断）→ 用户确认
          → sample-split + feature-screening → 一次建模数据确认
          → model-config → 用户选择
                ├─ bayesian-optimization ─┐
                └─ llm-optimization ──────┴→ model-review → END
```

当前试验版已接入独立 `data-read`、`eda-analysis`、`sample-split`、`feature-screening` 和 `model-config`。`model-config` 之后通过条件边在 `bayesian-optimization` 与 `llm-optimization` 中严格二选一，未选择的分支不会执行；两个分支最后汇合到 `model-review`。LLM 分支是真实的 LightGBM 迭代调参闭环：先训练 baseline，把当前参数和 Train/Validate/OOT 指标发送给 OpenAI 兼容接口，要求返回 JSON 参数补丁，经过参数白名单、类型和硬边界校验后重新训练；贝叶斯分支使用 Optuna/TPE 采样并逐 trial 训练 LightGBM。两种分支都记录 Train/Validate/OOT KS、综合模型质量分和拟合历史，并保存综合质量分最高的模型。质量分对 OOT KS 赋最高权重，同时惩罚 Train-Validate、Validate-OOT 的绝对差距和明显 OOT Score PSI；如要让 OOT 完全保持独立 holdout，可把优化配置的 `metric` 改为 `validate_ks`。`eda-analysis` 内部同时生成 EDA 指标和样本诊断摘要；`sample-split` 默认时间切分并排除最新不完整月份；`feature-screening` 只在 Train 计算区分度，在 Validate/OOT 计算稳定性。默认在项目的隐藏 `.langgraph` 目录写入一个 run 级 Arrow IPC 缓存，并在 checkpoint 中只保存缓存路径，因此后续节点不会重新读取源数据，也不会把 DataFrame 重复塞入每个 checkpoint；超大文件可把 data-read 的 `cache_data` 设为 `false`。

EDA 的 IV、KS、AUC 和 PSI 统一使用 `toad` 计算，Polars 负责文件读取、惰性扫描和结果落盘；字段画像中会标记 `metrics_backend: toad`。返回给 OpenCode 的内容是包含样本量、坏样本率、IV/KS、PSI、缺失、高相关和月份风险的精简摘要；完整 Top10、分箱和诊断明细仍保存在 `eda_conclusion.md` 与 HTML 报告中。

## 运行方式

```bash
PYTHONPATH=<skill-root>/scripts \
<python> -m cli start \
  --project-root <project-root> \
  --skill-root <skill-root>
```

OpenCode 集成时必须使用分离的 `start` / `resume` 调用，不要使用会读取终端
`input()` 的 `interactive`：

第一次调用：

```bash
PYTHONPATH=<skill-root>/scripts \
<python> -m cli start \
  --project-root <project-root> \
  --skill-root <skill-root> \
  --data <project-root>/data.csv
```

命令返回 `waiting_confirmation`、`run_id` 和当前节点摘要。OpenCode 将摘要展示给用户；用户确认后，第二次调用：

```bash
PYTHONPATH=<skill-root>/scripts \
<python> -m cli resume \
  --project-root <project-root> \
  --skill-root <skill-root> \
  --run-id <run-id> \
  --decision '{"approved":true}'
```

本地试验版使用文件持久化的 LangGraph checkpointer，支持跨进程恢复；生产环境可替换为 SQLite/PostgreSQL。`interactive` 仅用于直接在真实终端手工测试。

 LangGraph 状态只保存数据路径、配置路径、哈希、调参方式和一个内部 Arrow 缓存路径，不保存重复的 DataFrame 副本。data-read 只输出 Markdown 摘要和 JSON 摘要；EDA 默认只输出 `data_eda_report.html`、`eda_conclusion.md` 和 `eda_summary.json`；样本切分只输出 `split_manifest.json`；特征筛选输出 `feature_selection.csv`、`feature_selection_summary.md` 和 `feature_processing_manifest.json`；模型阶段只产生用户选中的优化分支产物，未选分支不会执行。LLM 训练执行器每完成一轮调用 `TrainingMonitor.record_iteration(...)`，统一写入 `outputs/<run_id>/modeling-monitor/run.json` 和 `iterations.jsonl`：每行是一轮参数、Train/Validate/OOT 指标、综合质量分、状态和可选拟合曲线，适合后续独立网页轮询读取。LLM 请求可通过 `timeout_seconds`、`max_retries_per_round` 和 `enable_thinking: false` 控制单轮耗时，并在 `run.json` 中写入当前请求/训练轮次。`model-review` 会读取该 JSONL，按综合质量分选择最终轮次，并生成 `outputs/<run_id>/model/model_report.html`；报告包含固定导航栏、模型结论卡、Baseline 对比、最佳模型拟合曲线和 ROC/PR 曲线、AUC、KS、Gini、PR-AUC、Score PSI、综合质量分、训练轮次与关键参数对比、月度稳定性、样本切分，以及入模变量的 IV、Gini、信息熵、缺失率、LightGBM Gain 重要性和 PSI，不再输出十分位和分数分布章节。评估会排除预测分数只有一个取值的退化 trial，避免把 AUC=0.5 错误显示成 KS=1。如果尚无迭代记录，会明确提示尚未训练，不伪造模型结果。需要逐表 CSV 时，开发者可显式调用 `run_eda(export_details=True)`。开发环境可使用内存 checkpointer，生产环境应替换为 SQLite/PostgreSQL。
