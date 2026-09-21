---
name: risk-modeling-pipeline
description: "仅在用户明确要求开始、继续或确认自动风控建模时使用。按读取数据、EDA、样本诊断、特征处理、模型配置训练和模型审查的顺序执行；普通问候、一般咨询、配置说明或仅打开建模面板时不要调用。"
---

# 自动化风控建模 Skill

这个 Skill 把一次风控建模任务组织成一条简洁的瀑布流程。Python 负责读取、统计、
分箱、切分、训练、评估和写文件；助手负责解释证据、提出建模建议和展示结果。
普通用户只需确认数据字段角色和节点结果，不需要逐项填写复杂 YAML。除 `data-read` 外，后续节点直接
使用工作区现有 YAML 或 Skill 默认参数执行；每个节点完成后返回中文总结、关键证据和文字化
参数修改建议，并等待用户确认结果后再继续。完整配置仍保留在工作区，供审计和高级用户按
文字指令调整。

## 1. 什么时候启动

只有用户明确表达以下意图时才启动：

- 开始、运行或继续自动建模；
- 生成建模任务；
- 确认当前节点配置并进入下一步；
- 按已选节点执行数据分析或模型训练。

以下操作不会启动流程：普通问候、询问模型或提供商、配置环境、打开 AI
建模模式、勾选节点但尚未点击“生成并发送建模任务”。

任务开始后，宿主会提供两个绝对路径：`project_root` 是用户工作区，
`engine_root` 是内置 Skill 根目录。不要把用户工作区当作引擎目录，也不要
要求用户把 Skill 源码复制到工作区。

## 2. 总体瀑布流程

```text
用户选择节点并生成任务
        │
        ▼
① data-read：读取数据、生成字段角色草案 ── 用户确认 Markdown 结果表
        │
        ▼
② eda-analysis：按已确认角色做 EDA ───── 只生成报告，自动进入样本诊断
        │
        ├── 用户选择 sample-diagnosis 时
        │       ▼
│   ③ sample-diagnosis：样本问题诊断 ── 默认参数，生成总结和建议，等待结果确认
        │
        ▼
④ feature-processing：统计、筛选并生成处理数据 ─ 执行并等待结果确认
        │
        ▼
⑤ model-config：展示建模决策摘要并等待确认 ─ 确认后训练调参
        │                                      │
        │                                      ├─ 调参方式三选一：none / Optuna / LLM
        │                                      └─ 评估、审查、报告和模型文件
        ▼
⑥ model-review：查看训练阶段生成的审查结果和改进建议
```

节点只按用户本次选择的白名单执行。最后一个已选节点完成后停止，
不得主动询问或调用未选节点。`training-tuning`、`tuning`、`review` 和
`model-report` 是引擎内部阶段；用户界面把训练和调参合并到 `model-config`，
把模型审查作为训练产物展示。宿主不能把它们再拆成第二次用户确认。

推荐交互采用“结果优先、配置可追溯”：基础模式只确认字段角色；EDA 只生成完整报告并自动进入样本诊断，
不在聊天区重复输出 EDA 明细；特征处理执行并展示证据、总结和建模建议，然后等待结果确认；模型配置先展示一张决策表并等待确认，确认后才开始训练调参；样本诊断使用默认参数并等待诊断结果确认。高级用户明确提出调整时，助手
才将文字指令转换为对应 YAML 修改并重新运行当前节点。这里的“大模型建议”是基于节点产物
结构化证据生成的解释和建议，不代表自动改数或自动改模型。

每个节点只调用一次并输出反馈卡片。进度事件中的 `summary` 是节点结果的唯一聊天展示源；
命令 JSON 返回值只用于状态和产物路径，不要再次复述同一段 `summary`。`data-read` 首次运行必须先展示引擎返回的
`summary` Markdown 表格（文件名、行列数、ID/日期/Y 字段、标签映射和特征数量），只等待
用户确认表格或提出文字修改；YAML 和哈希仅用于内部校验，不展示给用户。EDA 只生成报告并自动通过；样本诊断、特征处理和模型配置完成后等待结果确认，不要求用户填写 YAML；
用户如需调整，直接说明要修改的业务决策或参数，助手修改当前节点 YAML 后重跑。

### 配置修改边界（必须遵守）

节点执行后允许用户采纳建议、拒绝建议，或用自然语言提出调整要求。助手应将要求转换为
当前节点 YAML 中可配置的参数，说明修改后的关键字段和影响范围，再重新运行当前节点；除
`data-read` 外不追加配置确认门禁，但仍需用户确认节点结果。

- 所有修改只能写入用户工作区 `<project-root>/configs/` 下的 YAML 配置（当前节点的
  `configs/node_configs/<node-id>.yaml`，以及确有需要时的共享 `data_contract.yaml`）。
- 严禁修改、覆盖或临时打补丁到 `<engine-root>` 下的 Skill 源代码、脚本、提示词和模板。
- 每个参数必须经过节点自身的类型、取值范围和相互约束校验；非法或超出硬边界的要求应说明
  原因并要求用户改用合法值，不能静默截断或猜测。
- 如果用户的要求需要新增算法或改变引擎逻辑，而现有 YAML 没有对应参数，说明当前 Skill
  不支持该调整，并保留源代码不变；不要为了满足单次任务直接修改 Skill。
- 配置修改后必须重新生成结果摘要、配置哈希和审批状态；旧审批自动失效，不能沿用旧结果。

下游节点需要前置证据时，先执行已选范围内缺失的前置节点；如果用户没有选择
前置节点，应在任务生成前提示选择，而不是在执行中静默扩展白名单。

## 3. 每个节点怎么做

### 3.1 读取数据与契约校验（`data-read`）

目标是确定数据能否被后续节点正确使用，不做 EDA 和建模。

执行步骤：

1. 按明确的 `--data`、契约 `input_path` 或唯一数据文件顺序定位数据；CSV、
   Parquet 和 Excel 使用 Polars 读取。多个候选文件必须先让用户选择。
2. 读取 `configs/data_contract.yaml`，并应用
   `configs/node_configs/data-read.yaml` 中的编码、sheet、`id_col_nm`、
   `dt_col_nm` 和 `label_col_nm` 覆盖项。
3. 生成行列数、字段类型、主键唯一率、日期解析率、标签取值和特征列表。
4. 第一次调用只生成/补全内部 `data-read.yaml` 和确认摘要，返回
   `waiting_confirmation`；向用户只展示固定 Markdown 结果表，不展示 YAML 内容和 SHA-256。
5. 用户确认后，对同一节点调用 `workflow confirm data-read`，通过哈希校验后
   写入审批文件和读取结果。

用户必须确认数据文件、ID 及唯一性含义、观察日期、目标字段、好坏标签映射和
排除字段。主要产物为 `outputs/data-read/data_read_summary.json`（含固定 Markdown
表格）、`outputs/data-read/data_read_summary.md`、读取确认文件、
`configs/node_configs/data-read.yaml` 和 `configs/approvals/data-read.approval.json`。

### 3.2 EDA 分析（`eda-analysis`）

EDA 读取已确认的数据角色和当前数据，不设置第二次参数表单。它只回答数据和变量表现如何：
不切分 Train/Test/OOT、不应用 `class_weight`、不训练模型、不自动删除字段。EDA 完成后自动写入
报告审批标记并进入下一个已选节点；聊天区不输出长篇 EDA 摘要，完整内容直接查看 HTML/Excel 报告。

执行步骤：

1. 重新读取数据和 `data-read.yaml`，校验契约哈希及字段角色。
2. 计算分月样本总量、好坏样本量、坏账率和月度 PSI。
3. 计算每个变量的类型、缺失率、均值、标准差、IV、KS 和分箱统计。
4. 计算变量 PSI、相关性矩阵、月度 AUC/KS/Lift 及变量分布图。
5. 生成离线 HTML 主报告、Excel 明细、`eda_summary.json`、`eda_conclusion.md`
   和面向建模的中文摘要。

高 PSI、高相关、弱区分度、样本断层和最新月份过少都标记为“需关注”。这些
是后续决策证据，不是阻断条件。主要产物位于 `outputs/eda-analysis/`，推荐查看
`reports/data_eda_report.html`；明细包括 `column_profile.csv`、
`univariate_overview.csv`、`monthly_psi.csv`、`correlation_matrix.csv`、
`monthly_discrimination.csv` 和 `ks_bucket.csv`。

### 3.3 样本诊断（`sample-diagnosis`，可选）

该节点必须在 EDA 完成后执行。它使用 Skill 默认诊断参数，直接输出一张“样本诊断与建模风险提示”表，
不等待 YAML 配置确认，但必须等待诊断结果确认；不调用大模型，也不修改样本。表格包含 IV 达标占比、
PSI 有效/稳定占比、坏样本率、标签主键质量、缺失率、类别平衡和时间完整性，并在每行解释采用的阈值。

检查重复主键、缺失或非法标签、全空特征、高缺失样本、类别不平衡、月份坏账率
异常和最新月份完整性。诊断只产生证据和建议：不删除样本、不保存处理后数据、
不切分、不应用 `class_weight`，告警也不能阻止后续已选节点。

样本处理方法在后续 `model-config.yaml` 中按当前配置/defaults 直接采用，例如重复记录、缺失标签、
最新月份是否排除和 Train-only 类别权重；用户如需调整可直接提出文字指令。结果写入
`outputs/sample-diagnosis/modeling_risk_summary.json`、`sample_diagnostics.json` 和 `node_summary.json`。

### 3.4 特征预处理与筛选（`feature-processing`）

该节点自动读取工作区的 `feature-processing.yaml`（不存在时按 Skill 模板创建），
统计证据并立即生成处理数据；不要求普通用户确认配置，但等待结果确认后才进入下游节点。

执行步骤：

1. 重新读取已确认的数据和 EDA 结果；变量类型、缺失率、IV、KS、相关性和
   月度稳定性全部由代码计算。
2. 按变量类型生成预处理计划：数值变量缺失策略、类别变量编码/稀有类别、
   高基数和文本样字段处理。
3. 根据缺失率、`min_iv`、最大相关系数、最大 PSI 和稳定性阈值生成保留、剔除、
   待复核清单。
4. 按当前 YAML 自动应用筛选和预处理，生成 `processed_data.parquet`、字段去留清单
   和预处理计划；结果摘要必须用一张 Markdown 表说明输入/输出规模、保留字段、
   剔除字段（展示前 8 个及逐字段原因，字段过多时给出总数和完整清单路径）、
   待复核字段、筛选阈值、IV/KS/PSI 指标范围和变量类型处理方式。主要产物为
`feature_statistics.csv`、`feature_selection.csv`、`feature_selection_summary.md/json`、
`feature_preprocessing_plan.json`、`processed_data.parquet` 和
`feature_processing_report.html`。宿主大模型根据这些确定性证据输出总结和文字化参数修改
建议；节点本身不再生成或应用独立的配置建议 YAML。

该节点不做 Train/Test/OOT 样本切分，也不应用 class_weight；时间切分属于后续
`model-config` 的建模配置，不在本节点结果摘要中展示。

### 3.5 模型配置、训练与调参（`model-config`）

这是一个复合节点。第一次调用只读取当前模型配置并展示一张建模决策摘要表，包含样本时间
划分及原因、调参方式和模型选择（当前仅 LightGBM），不展开完整 YAML；节点随后暂停等待
用户确认。用户确认后，宿主调用 `workflow confirm model-config`，节点才在同一进程内完成
训练、调参、模型审查和报告交付，宿主不再单独调用 `training-tuning`。

节点检查 `data-read` 和 `feature-processing` 的审批哈希、已确认字段和
`configs/node_configs/model-config.yaml`。普通用户无需逐项确认样本处理、时间切分、
类别不平衡、LightGBM/Optuna 参数；引擎使用模板默认值并在摘要中说明实际方案和建议。
高级用户可编辑该 YAML 后重新运行；低层默认参数由引擎补齐。
开始训练前必须先让用户选择一种调参方式：`none`、`optuna` 或 `llm`，并写入
`tuning.method`；三种方式互斥，未选择时不得启动自动调参。

执行时的内部顺序：

1. 按确认的样本策略处理数据；Test/OOT 保持自然分布。
2. 按时间策略保留 OOT，再用 OOT 前月份作为 Test；Train 内按滚动扩展窗口
   做验证。Test/OOT 不参与调参决策。
3. 训练 LightGBM Baseline；根据 `tuning.method` 只选择一种调参方式：`optuna`
   在固定 Train-only 窗口和硬边界内试验参数，`llm` 从 Baseline 开始逐轮提出并重训，
   `none` 则不做自动调参。每个候选都记录参数、验证指标和最佳迭代轮次。
4. 用 `acceptance_metric` 和 `min_improvement` 比较 Baseline 与候选；未达到
   提升门槛时保留当前最优模型。
5. LLM 方式每轮做边界校验、重训和 Validate 指标比较，达到提升条件才接受；若
   Train/Validate KS gap 明显收窄且 Validate/OOT 仅小幅回落，也可按
   `guardrails.allow_gap_improvement` 保留为泛化改进候选。
   同时检查 Train/Validate 过拟合差距、Validate/OOT 退化和 OOT 相对 Baseline 的变化。
   默认 `run_all_rounds=true`，会用完配置的 10 轮预算；每轮都训练并记录，即使候选没有被接受，
   只有连续服务错误达到阈值、越界或无法完成训练时才提前结束。失败、越界、服务不可用或证书问题只记录原因，不使训练失败。
   OOT 只作为最终验证和护栏，LLM 永远不能直接用 OOT 选择参数。
6. 生成评估、模型审查、报告和模型文件。

### 3.6 模型审查（`model-review`）

模型审查使用训练阶段已经生成的真实结果，不重新训练。它汇总 Train/Test/OOT
的 AUC、KS、PR-AUC、Brier、Lift、校准、PSI、月度稳定性、随机种子波动和特征
重要性变化，判断过拟合、欠拟合、OOT 退化、排序性不足和样本支持不足，给出
带证据的改进建议。

当前审查结果由 `model-config` 复合节点生成到
`outputs/training-tuning/run/ai_model_review.json`，并写入模型报告。用户界面
的 `model-review` 只负责展示这些结果；没有独立的 `workflow model-review`
执行器，宿主不得把它当作新的训练命令。

## 4. 配置、确认和审批

```text
<project-root>/
├── data.csv 或其他数据文件
├── configs/
│   ├── data_contract.yaml                 # 全流程共享契约
│   ├── model_config.yaml                  # model-config 执行时生成的规范快照
│   ├── node_configs/
│   │   ├── data-read.yaml
│   │   ├── eda-analysis.yaml
│   │   ├── sample-diagnosis.yaml
│   │   ├── feature-processing.yaml
│   │   └── model-config.yaml
│   └── approvals/                          # 用户确认后的哈希审批
└── outputs/<node-id>/
```

模板从 Skill 的 `assets/` 按需复制到工作区，不要一次性复制所有模板。
`data_contract.yaml` 是跨节点共享的数据契约；节点 YAML 只放本节点参数。配置或
数据发生变化、前置审批失效时，必须重新执行对应节点并重新确认。

`data-read` 确认字段角色和读取配置；`eda-analysis` 自动通过报告产物；`sample-diagnosis`、
`feature-processing` 完成后返回 `waiting_confirmation`，展示总结和文字化参数建议，等待
用户确认结果；`model-config` 则在训练前返回一张决策表并等待确认。用户确认后，宿主调用
`workflow confirm <node-id>` 进入下一个节点；如需调整，用户直接提出文字指令，助手修改
当前 YAML 后重跑。模型配置表只能展示一次，JSON 状态中的同名摘要不得再次渲染。

## 5. OpenCode 调用方式

命令中的尖括号是宿主在执行前替换的变量，不是要原样输入终端的文本：

| 模板变量 | 来源 | 示例 |
|---|---|---|
| `<project-root>` | 当前 OpenCode 项目的绝对工作区路径 | `/Users/meixiaohan/Desktop/v2` |
| `<engine-root>` | 已安装 Skill 的绝对根目录 | `/Users/meixiaohan/Desktop/model_skills` |
| `<node-id>` | 流程面板本次选中的、当前轮到执行的节点 ID | `data-read`、`feature-processing`、`model-config` |

前端在用户点击“生成并发送建模任务”时，会把所选节点的 ID 写入任务白名单，
例如 `data-read,eda-analysis,feature-processing`。OpenCode 每次只取白名单中
当前节点的一个 ID，按固定顺序执行；节点完成后返回结果确认事件，用户确认后才取下一个。它不从
中文节点名称猜 ID，也不能把 `training-tuning` 或 `llm-tuning` 自行追加到白名单。

节点必须作为独立进程调用，固定格式如下：

```bash
PYTHONPATH=<engine-root>/risk-modeling-pipeline/scripts \
<engine-root>/.venv/bin/python -m workflow <node-id> \
  --project-root <project-root> \
  --engine-root <engine-root> \
  --output-dir <project-root>/outputs/<node-id>
```

兼容旧版脚本时，`eda` 也可作为 `eda-analysis` 的短别名；引擎会将其规范化为
`eda-analysis` 并写入相同的输出目录和进度事件。新的提示词和集成代码仍应优先
使用完整节点 ID `eda-analysis`，避免把别名误当成新的流程节点。

用户确认节点结果后，对同一节点使用（`data-read` 代表配置确认，其他节点代表结果确认）：

```bash
PYTHONPATH=<engine-root>/risk-modeling-pipeline/scripts \
<engine-root>/.venv/bin/python -m workflow confirm <node-id> \
  --project-root <project-root> \
  --engine-root <engine-root> \
  --output-dir <project-root>/outputs/<node-id>
```

禁止使用 `<engine-root>/scripts`、`risk-modeling-pipeline/.venv/bin/python`、
`risk-modeling-pipeline/scripts/.venv/bin/python` 和 `*/model_skills/scripts`。
不要用旧的 `--mode all`、`--mode prepare`、`--mode check` 或批处理 runner
代替用户选中的独立节点；环境诊断除非用户明确要求也不要调用 `--mode check`。

每条命令只能出现一次 `--project-root`、`--engine-root` 和 `--output-dir`。
不要通过 `ls`、`find`、`pwd` 猜测路径。相同命令已经成功返回后不要重试。

例如当前节点是 `feature-processing`，普通执行命令应是：

```bash
PYTHONPATH="/abs/model_skills/risk-modeling-pipeline/scripts" \
"/abs/model_skills/.venv/bin/python" -m workflow feature-processing \
  --project-root "/abs/user-project" \
  --engine-root "/abs/model_skills" \
  --output-dir "/abs/user-project/outputs/feature-processing"
```

这里的 `feature-processing` 来自流程面板 ID；`/abs/model_skills` 和
`/abs/user-project` 也必须由宿主提供真实绝对路径。不能把上面的 `/abs/...`
或尖括号内容直接复制执行。

第一次调用节点时不要追加确认参数。`data-read` 返回 `waiting_confirmation`
后，先原样展示返回的 summary 表格，并停止等待用户确认或修改；不要展示 data-read YAML；其他节点返回 `waiting_confirmation`
后展示结果摘要并停止。用户明确确认后调用 `workflow confirm <node-id>`；该命令对
`data-read` 使用配置确认，对其他节点使用结果确认。`--confirm-config` 仅作为
`data-read` 和旧集成的兼容参数保留。

## 6. 输出、进度和用户回复

每个节点写入 `run_events.jsonl`（完整事件流）、`run_state.json`（可重连快照）、
`node_summary.json`（确定性中文摘要）以及节点专属 JSON/CSV/Parquet/HTML/Excel
产物。训练阶段还会为每个 Optuna Trial 和每轮 LLM 调参写入 `experiment` 事件，
前端应把它们合并为一张可展开的参数历史表。

节点完成后的中文回复遵循对应模板：EDA 只报告完成情况和报告路径，不在聊天区复制摘要表；
样本诊断输出一张包含 IV/PSI/坏样本率等指标及阈值解释的风险表；特征处理输出保留/剔除/待复核字段及阈值影响；模型配置
训练输出最终候选、Train/Test/OOT 指标、调参接受原因、审查结论和交付路径。

完整图表和明细通过 HTML/Excel/固定产物路径查看。摘要数值必须来自本次运行文件
或进度事件，不能使用示例值或模型猜测。数据质量、PSI、类别不平衡和月份波动
统一写成“需关注”，不写成“无法建模”或自动阻断后续已选节点。

回复模板：

- [EDA 回复模板](references/eda-response-template.md)
- [data-read 回复模板](references/data-read-response-template.md)
- [样本诊断回复模板](references/sample-diagnosis-response-template.md)
- [模板索引](references/response-templates.md)

## 7. 建模规则和可选 AI 能力

- 类型字典、缺失处理、IV、稳定性和冗余筛选只在 Train 上拟合；Test/OOT 使用
  冻结后的处理规则。
- 默认调参目标是 Train-only 滚动验证 KS，并扣除过拟合 gap 和折间波动惩罚；
  可配置为 AUC。OOT 只用于最终评估和审查。
- `class_weight` 只在确认的 Train 配置中生效，不能改变 Test/OOT 的自然分布。
- Toad 默认计算 IV、KS 和 PSI；指标后端会记录在输出表中，缺少 Toad 时才使用
  兼容回退并给出警告。
- `feature-processing.parameters.llm` 只能读取 EDA 聚合证据并生成受边界约束的
  建议 YAML；建议不会自动生效。
- `tuning.method` 控制互斥的调参方式：`none` 只训练 Baseline，`optuna` 只运行 Optuna，
  `llm` 只运行 LLM 逐轮调参。`tuning.llm` 是 `model-config` 内部的 LLM 实现，不是独立节点；
  模板默认选择 `llm`；选择 `llm` 时从 Baseline 开始，不会先运行 Optuna。配置中的
  `api_key`、`base_url`、`model`、`max_rounds`、`min_improvement`、`stopping` 和 TLS
  选项必须进入调参历史；临时超时或服务错误会按 `request_retries` 有限重试，重试失败只记录并继续下一轮，连续失败达到 `max_consecutive_provider_errors` 才停止。`stopping.targets`
  定义 Validate/OOT 目标，`stopping.guardrails` 定义过拟合、时间退化和 OOT 样本量
  护栏；OOT 坏样本不足时只提示，不作为硬门槛。LLM 每轮必须同时看到当前指标、
  历史参数及被拒绝原因；候选比较使用固定随机种子和冻结的 Validate 集，避免把
  随机波动误判为参数提升。Validate 是主接受指标，OOT 只做最终验证和退化护栏；
  OOT 单独变好不能绕过 Validate 接受门槛。
- 模型交付至少包括 `lightgbm_model.pkl`、`lightgbm_model.txt`、
  `model_bundle.pkl`；PMML 转换不可用时写入 `pmml_export_status.json`。

## 8. 需要深入细节时读取的资料

- [AI 协作边界](references/ai-workflow.md)：确认门禁、助手职责和禁止事项；
- [配置参数说明](references/configuration.md)：契约、样本、特征和模型参数；
- [节点配置模板](references/node-configs.md)：各 YAML 的字段结构；
- [进度事件协议](references/progress-events.md)：OpenCode 前端接入和实验事件；
- [项目架构](references/architecture.md)：目录结构、脚本入口和外部调用方式。
