# risk-modeling-pipeline-langgraph

独立实现的 LangGraph 版风控建模 Skill 试验目录，不依赖旧版 `risk-modeling-pipeline`。当前已接入 `data-read`、包含样本诊断的 `eda-analysis`、`sample-split` 和 `feature-screening`，用于验证：

1. 使用本目录的 Polars loader、契约解析和字段角色推断读取数据，并使用 Lazy/Streaming 方式生成 EDA 结果；
2. 通过 LangGraph `interrupt()` 暂停并返回待确认内容；
3. 将 checkpoint 持久化到项目目录，支持 OpenCode 通过 `start` / `resume` 跨进程恢复。

## OpenCode 调用提示词

将下面的提示词复制到 OpenCode，并把项目目录和数据文件替换成实际路径：

```text
请调用 risk-modeling-pipeline-langgraph Skill，开始一次自动建模数据分析流程。

项目目录：
v2

数据文件：
v2/data123.csv

本次只执行以下节点：
1. data-read
2. eda-analysis（包含样本诊断）
3. sample-split
4. feature-screening
5. model-config
6. bayesian-optimization 或 llm-optimization（二选一）

执行要求：
- 使用 Skill 的 cli start 命令，不要使用 interactive；
- data-read 完成后返回 Markdown 数据摘要，并等待我确认；
- 我确认后，使用相同 run_id 调用 cli resume；
- eda-analysis 同时生成 EDA 指标和样本诊断结果；
- EDA 完成后返回诊断摘要、建模建议和报告路径，并等待我确认；
- sample-split 默认按时间切分 Train/Validate/OOT，自动排除最新不完整月份但不删除原始数据；
- feature-screening 只在 Train 计算 IV/KS/AUC，在 Validate/OOT 计算 PSI 和稳定性；
- 样本切分和特征筛选合并为一次“建模数据确认”，确认后进入 model-config；model-config 提供贝叶斯优化和 LLM 调参两个互斥分支，用户只能选择一个；默认把数据写入一个隐藏的 run 级 Arrow 缓存，LangGraph checkpoint 只保存引用，不重新读取源数据，也不生成 Train/Validate/OOT Parquet；超大文件可在 data-read 配置中关闭 `cache_data`，此时才按清单重新读取原始数据；
- 为避免产物泛滥，data-read 只保留 Markdown/JSON 摘要；EDA 默认只保留 HTML 报告、Markdown 结论和 JSON 摘要；样本切分只保留 `split_manifest.json`，特征筛选保留 `feature_selection.csv`、`feature_selection_summary.md` 与 `feature_processing_manifest.json`。详细 EDA 明细已内嵌到 HTML 报告，必要时再显式导出 CSV；
- 训练阶段不按 trial 创建零散文件，所有轮次追加写入 `outputs/<run_id>/modeling-monitor/iterations.jsonl`，运行元数据写入 `run.json`，后续建模监控网页可直接轮询 JSONL；
- LLM 分支结束后将综合模型质量分最高的 LightGBM 保存为 `outputs/<run_id>/model/model.pkl`。质量分优先考虑 OOT KS，并惩罚 Train/Validate/OOT KS 差距；模型审查摘要同时返回该路径；
- 模型审查会基于 `modeling-monitor/iterations.jsonl` 自动生成 `outputs/<run_id>/model/model_report.html`，包含固定导航栏、模型结论卡、最佳参数、Baseline 与最佳轮次对比、最佳模型的拟合曲线和 ROC/PR 曲线、Train/Validate/OOT 的 AUC、KS、Gini、PR-AUC、Score PSI、综合质量分、每轮候选参数对比、月度稳定性、样本切分，以及入模变量的 IV、Gini、信息熵、缺失率、LightGBM Gain 重要性和 PSI；不再输出缺乏决策价值的十分位和分数分布章节。评估会排除预测分数只有一个取值的退化 trial，避免把 AUC=0.5 错误显示成 KS=1；即使调参中途超时，也只报告已经落盘的真实轮次；
- LLM 模板已内置 Qwen 兼容接口地址、`qwen3.6-flash` 默认模型和当前本地测试 API Key；发布或提交 Git 前必须删除 `api_key` 并轮换密钥，也可改回使用 `LLM_TUNING_API_KEY`；
- LLM 调参分支会真实训练 LightGBM 基线并逐轮重训候选参数；每轮由大模型读取当前参数和 Train/Validate/OOT 指标，返回 JSON 参数提议，经白名单和硬边界校验后再训练。最终 `model-review` 从迭代日志按综合质量分选择模型，并汇总 Train/Validate/OOT KS；如果配置缺失或没有训练迭代，会明确提示原因，不会伪造模型结果；

每轮日志的结构固定为：

```json
{
  "iteration": 1,
  "method": "bayesian",
  "parameters": {"learning_rate": 0.03},
  "metrics": {
    "train": {"ks": 0.42, "auc": 0.71},
    "validate": {"ks": 0.40, "auc": 0.69},
    "oot": {"ks": 0.35, "auc": 0.66}
  },
  "fit_history": {"train": [], "validate": []},
  "status": "complete",
  "accepted": true
}
```
- 进入模型阶段后，必须先等待 model-config 的调参方式选择；未选择前不要执行任何优化分支；
- 不要自动跳过确认，也不要重复输出同一个节点结果。
```

第一次看到 `data-read` 摘要后回复：

```text
确认 data-read，继续执行 EDA 和样本诊断。
```

看到 EDA/样本诊断摘要后回复：

```text
确认 EDA 和样本诊断，继续样本切分。
```

看到样本切分和特征筛选合并摘要后回复：

```text
确认建模数据切分和特征筛选，进入模型配置。
```

看到模型配置摘要后，必须明确选择一种调参方式：

```text
确认模型配置，选择 bayesian。
```

或：

```text
确认模型配置，选择 llm。
```

对应的 `resume` 决策 JSON 为 `{"approved":true,"tuning_method":"bayesian"}` 或 `{"approved":true,"tuning_method":"llm"}`。未选择有效方法时会停留在 model-config，不会触发任何调参分支。

## 本地试运行

先安装依赖：

```bash
<python> -m pip install -r \
  risk-modeling-pipeline-langgraph/requirements.txt
```

OpenCode 集成测试使用一次 `start` 和多次 `resume` 调用，每个节点确认后恢复一次：

```bash
PYTHONPATH=risk-modeling-pipeline-langgraph/scripts \
<python> -m cli start \
  --project-root <project-root> \
  --skill-root <skill-root> \
  --data <project-root>/data.csv
```

命令返回 `waiting_confirmation` 和 `run_id`。OpenCode 将摘要展示给用户，用户确认后调用：

```bash
PYTHONPATH=risk-modeling-pipeline-langgraph/scripts \
<python> -m cli resume \
  --project-root <project-root> \
  --skill-root <skill-root> \
  --run-id <run-id> \
  --decision '{"approved":true}'
```

`interactive` 仅用于直接在真实终端手工测试；EDA 的 IV、KS、AUC、PSI 使用 `toad` 计算。

报告展示约定：EDA 的浮点指标统一保留 5 位小数；比例字段在 HTML/Markdown 中以 5 位小数百分比展示。用户表头使用中文，保留 KS、PSI、STD 等标准指标缩写；仅保留的特征筛选 CSV 使用稳定的英文列名，便于程序读取。

旧版 `risk-modeling-pipeline` 保持不变；当前流程按 `data-read → eda-analysis → sample-split + feature-screening → model-config → bayesian-optimization/llm-optimization` 执行，两个调参分支严格互斥。LLM 分支和贝叶斯分支都已接入真实 LightGBM 训练与迭代监控：前者由大模型返回 JSON 参数，后者使用 Optuna/TPE 搜索参数。

## 独立训练监控网页

本目录还提供一个与 OpenCode、LangGraph 图和建模执行完全解耦的只读网页，用于查看已经生成的训练日志。它不会启动训练、修改配置或调用模型服务：

```bash
python monitor_web/serve.py
```

浏览器打开 <http://127.0.0.1:8765/index.html>，选择 `outputs/<run_id>/modeling-monitor/iterations.jsonl`。页面会每秒重新读取文件，展示所有轮次的 Train/Validate/OOT 指标、参数、接受状态和 `fit_history` 拟合曲线。Chrome/Edge 支持持续读取文件句柄；其他浏览器可重新选择文件刷新快照。
