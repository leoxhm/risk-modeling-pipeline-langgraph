# core 目录约定

`core` 只放每个节点的确定性业务实现，不包含 LangGraph API：

| 目录 | 职责 |
|---|---|
| `data_read` | Polars loader、契约解析、字段角色推断 |
| `eda_analysis` | 懒扫描、质量统计、IV/KS/PSI、报告数据 |
| `sample_diagnosis` | 重复、标签、缺失、倾斜和时间风险诊断 |
| `sample_split` | 按时间生成 Train/Validate/OOT 规则，排除不完整月份并输出切分清单（不落盘 Parquet） |
| `feature_screening` | Train 区分度、Validate/OOT PSI、相关性筛选和字段清单（不落盘 Parquet） |
| `feature_processing` | 后续类型处理、缺失策略和转换（预留） |
| `modeling` | LightGBM 基础模型、Optuna/TPE 贝叶斯优化、LLM JSON 调参和评估 |
| `model_review` | 过拟合、稳定性、OOT 和改进建议 |

每个目录都可以独立编写单元测试；LangGraph 节点只调用对应 core 函数。

## 状态与产物约定

数据读取只执行一次。LangGraph 将本次数据写入 `outputs/.langgraph/<run_id>.data-cache.arrow`，checkpoint 只记录这个路径；EDA、样本切分和特征筛选都从该缓存读取，不重新解析源 CSV/Excel。`cache_data: false` 时才允许后续节点按原始数据路径重新读取。

默认产物保持最小集合：data-read 输出 Markdown/JSON 摘要，EDA 输出 HTML 报告、Markdown 结论和 JSON 摘要，sample-split 输出 `split_manifest.json`，feature-screening 输出 `feature_selection.csv`、`feature_selection_summary.md` 和 `feature_processing_manifest.json`。详细 EDA 表格直接内嵌在 HTML 报告中，不默认写出十余个 CSV。

训练监控统一写入 `outputs/<run_id>/modeling-monitor/run.json` 和 `iterations.jsonl`。每轮只追加一行 JSON，包含参数、Train/Validate/OOT 指标、接受状态和可选 `fit_history`，前端可以按文件偏移量实时读取。

模型阶段由 `model-config` 选择一个互斥分支：`core/modeling/bayesian_tuner.py` 使用 Optuna/TPE 逐 trial 采样和训练，`core/modeling/llm_tuner.py` 使用 OpenAI 兼容接口返回 JSON 参数补丁后重训；两者默认使用 `core/modeling/evaluation.py` 的综合质量分选择候选：OOT KS 权重 45%，Validate KS 30%，Train KS 25%，并惩罚两个分区差距和明显 OOT Score PSI；最佳 LightGBM 保存到 `outputs/<run_id>/model/model.pkl`。`core/modeling/model_report.py` 在模型审查阶段从监控日志生成离线 `model_report.html`，包含 AUC、KS、Gini、Train/Validate/OOT 稳定性差距、综合质量分，以及入模变量的 IV、信息熵、缺失率、Gain 重要性和 PSI；不会因为缺少某轮数据而伪造指标，也不输出低价值的十分位章节。

```aiignore
请调用 risk-modeling-pipeline-langgraph Skill，开始一次自动建模数据分析流程。

项目目录：
/Users/meixiaohan/Desktop/v2

数据文件：
/Users/meixiaohan/Desktop/v2/data123.csv

本次只执行以下节点：

1. data-read
2. eda-analysis

执行要求：

- 使用 Skill 的 `cli start` 命令，不要使用 `interactive`
- data-read 完成后，只返回 Markdown 数据摘要表，并等待我确认
- 我确认后，使用相同 run_id 调用 `cli resume`
- 然后执行 eda-analysis
- EDA 完成后返回摘要和报告路径，并等待我确认
- 不要执行 sample-diagnosis、feature-processing、model-config、training-tuning 或 model-review
- 不要自动跳过确认，也不要重复输出同一个节点结果
```
