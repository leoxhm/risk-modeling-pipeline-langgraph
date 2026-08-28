# 风控自动建模 Skill

这是一个面向结构化风控数据的自动化建模项目。项目使用 Polars 完成数据读取、画像与清理，以 LightGBM 为主要模型、Optuna TPE 为自动调参器，并通过 Codex Skill 提供“AI 建议 + 用户确认 + 确定性执行”的自然语言入口。

部署到新环境并启动 OpenCode Web 前端，请参阅 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 当前运行流程

当前项目分为三层：Skill 负责对话与确认门禁，AI 辅助层负责诊断、字段语义建议、实验规划和模型红队审查，Python 建模引擎负责可复现的数据计算、调参、训练和落盘。一次完整运行必须按以下流程执行。

部署依赖分为两档：`requirements.txt` 是建模引擎运行时依赖；`requirements-full.txt` 在此基础上增加 JupyterLab 内核和测试工具，适合 Conda/JupyterLab 环境。

用户只需要理解下面八个主节点；画像、字段语义审查、清理、调参和报告导出等是节点内部子步骤：

```text
读取数据与契约校验
        ↓
EDA分析 → 反馈并确认
        ↓
样本诊断 → 反馈并确认
        ↓
特征预处理与筛选 → 反馈并确认
        ↓
生成模型配置 → 反馈并确认
        ↓
模型训练与调参 → 反馈并确认结果
        ↓
模型审查
        ↓
报告与模型交付
```

确认等待由节点状态 `waiting_confirmation` 表示，不再额外显示“用户确认”节点。Python 引擎仍保留详细内部事件，供审计、日志和故障定位使用。

### 1. 发起任务

可以通过自然语言调用 Skill，也可以直接执行统一命令行入口：

```text
用户 / Codex / OpenCode
        ↓
risk-modeling-pipeline/scripts/workflow/__main__.py
```

入口支持六种模式：

- `check`：检查 Python 依赖和项目核心文件。
- `prepare`：只读分析和 Train-only 特征预览，不修改原始数据、不训练；生成 AI 辅助材料与待确认清单。
- `approve`：在用户明确确认后生成带文件哈希的批准清单。
- `eda`：运行数据清理、探索分析并生成 EDA 报告。
- `model`：运行样本切分、特征筛选、LightGBM 训练与模型报告。
- `all`：在同一个运行目录中依次执行 EDA 和建模。

### 2. 预检与 AI 辅助建议

先运行 `prepare`。此阶段只生成只读证据、不会修改原始数据、训练模型或创建批准文件，会输出：

- `column_profile.csv`：确定性字段画像。
- `ai_preflight.json`：样本、时间窗口、缺失率和 OOT 支撑度诊断。
- `ai_schema_proposal.json`：主键、日期、目标、特征及泄漏风险建议。
- `ai_experiment_plan.json`：基线、调参、比较和停止条件建议。
- `feature_proposal.json`、`feature_preprocessing.csv`、`feature_selection.csv`：Train-only 特征类型、编码、去留、IV、相关性和稳定性证据。
- `model_config_proposal.yaml`：根据当前配置生成、等待用户确认的模型参数快照。
- `confirmation_request.json`：待确认角色、阻断项和输入文件 SHA-256。

特征预览是只读建议，标记为 `sample_treatment_applied=false`。如果用户在样本诊断阶段修改了重复、缺失标签、最新月份或类别不平衡处理方式，必须重新运行 `prepare`，再确认特征结果。

建模执行完成后，`ai_model_review.json` 会汇总模型诊断结论、规则阈值、原始指标证据和优先级改进建议，覆盖 Train/Test AUC 与 KS gap、欠拟合、Test→OOT 下降、分数 PSI、Lift 排序、月度稳定性、早停和特征重要性集中度；同样内容会写入 `model_report.xlsx` 的 `13.诊断建议` 工作表。

### 3. 用户确认与批准门禁

实际计算前，以下信息必须由用户明确确认：

- 主键字段及其唯一性含义。
- 观察日期字段及时间窗口。
- 目标字段、好坏标签映射。
- 不应入模的泄漏字段、结果字段和人工排除字段。
- 字符变量策略、高基数处理方式和稳定性异常是复核还是剔除。
- OOT 月份数、Test 比例、训练模式、调参预算和核心参数。

大模型只能提出建议，不能替用户决定目标含义、泄漏字段或接受阻断风险。若存在阻断项，用户需逐项确认；`approve` 才会生成 `approval_manifest.json`。正式执行会重新计算数据、契约和模型配置的哈希，任一文件发生变化都必须重新 `prepare` 和确认。

### 4. 读取数据与契约校验

`loader` 使用 Polars 读取 CSV、Excel 或 Parquet，然后由 `contract` 完成：

- 必需字段存在性检查。
- 主键唯一率检查。
- 日期可解析率检查。
- 二分类标签和好坏样本映射校验。
- 最终候选特征列确认。

契约未通过时立即停止，不会继续清理或训练。

### 5. EDA 分支

`eda` 或 `all` 模式会执行：

```mermaid
flowchart LR
    A["去重、日期标准化与特征清理"] --> B["原始数据探索"]
    B --> C["字段质量与单变量分析"]
    C --> D["IV / WOE / PSI / 相关性"]
    D --> E["CSV、Parquet 与 EDA Excel 报告"]
```

清理过程会保留字段去留决策和原因，不会只输出一份经过处理但无法追溯的数据。

### 6. 建模分支

`model` 或 `all` 模式会执行：

```mermaid
flowchart LR
    A["行级清理：日期、去重"] --> B["Train / Test / OOT 切分"]
    B --> C["Train 拟合变量类型与类别字典"]
    C --> D["Train 质量、IV、稳定性和相关性筛选"]
    D --> E["将冻结规则应用到 Test / OOT"]
    E --> F["可选：Train 内分层 CV + Optuna TPE"]
    F --> G["训练固定参数基线与最终模型"]
    G --> H["Test 最终候选比较与早停"]
    H --> I["OOT 最终时间外验证"]
    I --> J["AI 红队审查、评分与建模报告"]
```

无论是否调参，都会保留基线模型。默认情况下 Optuna 只使用 Train 内固定分层折，以平均验证 KS 减去过拟合差距和折间波动惩罚作为目标；设置 `tuning.objective_metric: auc` 可切换为 AUC。训练过程每轮同时记录 AUC/KS，并按 `model.early_stopping_metric`（默认 KS）在 Test 上早停。OOT 只做最后的时间外验证，不参与特征筛选、候选选择、阈值选择或调参。评估输出 AUC、KS、PSI、Lift、月度表现、分箱分析和 AI 红队风险清单。

### 7. 产物落盘与结果返回

每次运行建议使用独立的输出目录。程序会保留：

- 本次使用的数据契约和模型配置。
- 清理数据、切分摘要、特征决策和评估明细。
- LightGBM 模型文件、全样本评分数据和训练历史。
- Optuna SQLite study、全部 trial、CV 折指标、最优参数和候选比较。
- AI 预检、字段建议、实验计划及模型红队审查结果。
- `data_eda_report.xlsx` 和 `model_report.xlsx`。
- 包含输出路径、入模特征数和核心指标的 JSON 运行结果。
- `run_events.jsonl`：节点状态的追加事件流。
- `run_state.json`：可供界面刷新和断线恢复的最新节点快照。

建模命令的 JSON 返回值也包含 `model_review` 摘要（状态、阻断/警告数量、建议数量和 `ai_model_review.json` 路径），前端可以直接把诊断结果回填到“模型审查”节点。

报告和指标用于辅助评审，不会自动将模型标记为可上生产。

节点完成状态由 Python 引擎依据函数返回值和产物存在性生成，而不是由大模型推测。右侧流程图只显示八个主节点，内部事件会归并到对应主节点，并保留执行摘要、耗时、运行 ID 和产物路径。需要把多次命令关联为同一前端任务时，为 `prepare`、`approve` 和正式执行传入相同的 `--run-id`。

当前流程支持：

- CSV、Excel、Parquet 数据读取
- 字段画像与字段角色确认
- AI 入模前诊断、字段语义/泄漏建议和实验规划
- 数据契约校验
- 数据清理与探索性分析（EDA）
- Train、Test、OOT 数据切分
- Train-only 变量类型识别、字符变量编码和冻结转换
- 基于 Train 的缺失率、IV、月度 PSI 与 Spearman 冗余筛选
- LightGBM 二分类模型训练与早停
- Optuna TPE 自动调参与基线/调参候选比较
- AUC、KS、PSI、Lift、月度表现评估
- 基于证据的模型诊断：过拟合/欠拟合、时间外泛化、分数漂移、Lift 排序和月度稳定性
- EDA Excel 报告与模型 Excel 报告
- 模型文件、评分明细和可复现配置留存
- 哈希绑定的用户确认门禁与 AI 模型红队审查

## 设计原则

项目将大模型和确定性程序分开：

- 大模型负责理解需求、解释字段、提出字段角色建议、提示风险和解读报告。
- Python 程序负责数据计算、样本切分、特征筛选、调参、模型训练、指标计算和文件哈希校验。
- 目标字段含义、好坏标签、观察日期、主键、排除字段和 OOT 策略必须由用户确认。
- OOT 数据不得参与特征筛选、早停、阈值选择或参数调优。
- AI 不得代替用户生成确认，自动化结果也不得直接等同于可上线结论。

## 项目架构

部署时，`model_skills` 是 OpenCode 服务端内置的建模引擎和 Skill 资源；用户当前项目只是数据工作区，不需要复制 Skill 引擎源码。

```text
model_skills/                     # OpenCode 内置引擎
├── risk-modeling-pipeline/      # Codex Skill + Python engine
│   ├── SKILL.md                 # Skill 核心工作流
│   ├── agents/openai.yaml       # Skill 展示信息和默认提示词
│   ├── assets/                  # 配置模板和依赖清单
│   ├── references/              # AI 门禁、配置与迁移参考
│   └── scripts/                 # Python 引擎源码和 workflow 入口
│       ├── workflow/            # 节点定义、配置、独立节点和兼容 runner
│       ├── ai/ data/ eda/       # AI、读取、EDA 模块
│       ├── preprocessing/       # 样本和特征处理
│       ├── modeling/ metrics/   # 建模、指标和调参
│       ├── reporting/ logger/   # 报表和日志
│       └── progress.py          # 进度事件
└── requirements.txt             # Python 依赖

user_workspace/
├── configs/                      # 用户确认的运行配置
│   └── node_configs/             # 每个节点独立的可调整参数文件
├── data.csv                      # 用户数据
└── outputs/
    └── run_001/
        ├── models/               # pkl、原生 LightGBM、可选 PMML
        ├── reports/              # EDA/建模 Excel 报告
        └── artifacts/            # 清洗数据、指标明细、配置、日志和审计证据
```

## 环境安装

推荐使用 Python 3.11：

```bash
conda create -n risk-modeling python=3.11 -y
conda activate risk-modeling
cd "/Users/meixiaohan/Desktop/自动化建模工具 参考/model_skills"
python -m pip install -r requirements.txt
```

Excel 报表由 Python `XlsxWriter` 生成，不依赖 Node.js、OpenCode 私有运行时或 `@oai/artifact-tool`。

## v1 数据与 EDA 流程

v1 先开放数据读取、EDA 和样本诊断三个有效节点：

```text
data-read（读取数据与契约校验）
        ↓
eda-analysis（界面显示为“EDA分析”）
        ↓
sample-diagnosis（界面显示为“样本诊断”，按需选择）
```

读取节点采用“两阶段读取”：第一次使用 `data-read.yaml` 模板中的文件读取参数加载原始表，并根据数据契约和字段画像规则补全 `id_col_nm`、`dt_col_nm`、`label_col_nm`；OpenCode 也可以在这一阶段给出建议。系统会暂停，用户可以修改 YAML；修改后重新执行节点，刷新字段证据并生成新的确认请求。正式执行时重新读取并校验 YAML，按最终的编码、Sheet、类型推断和字段角色配置继续执行，然后进入 EDA。EDA 完成后，只有用户选择 `sample-diagnosis` 才会读取同一个 `configs/node_configs/sample-diagnosis.yaml`，诊断样本倾斜、月份异常、重复主键、缺失标签和最新月份完整性；用户确认后只记录样本处理策略，不修改或保存处理后的数据。特征处理节点随后先生成缺失率、IV、KS、相关性和稳定性证据，用户确认 `feature-processing.yaml` 中的筛选阈值及变量类型处理方法后，才生成处理数据。每个节点的参数位于 `configs/node_configs/`。模型配置、训练、审查和交付节点暂时只保留配置模板，待数据/EDA/特征链路稳定后再开启。

OpenCode 逐节点执行时，每个节点都是独立进程，不依赖上一个节点的 Python 内存状态。节点入口位于 `workflow/nodes/`，例如：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow data-read \
  --project-root . --engine-root . --data data.csv \
  --data-contract configs/data_contract.yaml \
  --node-config-dir configs/node_configs --output-dir outputs/data-read

PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow eda-analysis \
  --project-root . --engine-root . --data data.csv \
  --data-contract configs/data_contract.yaml \
  --node-config-dir configs/node_configs --output-dir outputs/eda-analysis

PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow sample-diagnosis \
  --project-root . --engine-root . --data data.csv \
  --data-contract configs/data_contract.yaml \
  --node-config-dir configs/node_configs --output-dir outputs/sample-diagnosis
```

每个节点都会重新读取 `data-read.yaml` 的数据读取参数，同时读取自身 YAML 并把产物写入自己的输出目录。`python -m workflow --mode ...` 仍保留给批量 `prepare/approve/eda/model/all` 兼容流程。

首次调用节点只负责复制模板、生成建议和确认摘要，会返回
`awaiting_user_confirmation`；用户确认 YAML 后，再在同一命令上追加
`--confirm-config` 执行确认后的节点逻辑。

在 Linux/OpenCode 中只需要安装 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

依赖包括 Polars、Toad、XlsxWriter、LightGBM 和 Optuna。`--node-executable` 参数保留用于兼容旧命令，但新版报表生成不会调用 Node。

检查环境：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow \
  --mode check \
  --engine-root .
```

当输出中的 `ready` 为 `true` 时，Python 依赖和核心项目文件已经就绪。

## 运行前配置

说明：`risk-modeling-pipeline/assets/` 保存可复制的通用模板；根目录的
`configs/` 是本地 `data.csv` 的测试工作区配置。部署或新建用户工作区时，
应从 `assets/` 复制模板到用户项目的 `configs/` 后再编辑确认。

### 数据契约

编辑 `configs/data_contract.yaml`：

```yaml
data:
  # 可选；为空时自动发现工作区根目录下唯一的数据文件
  input_path: null

schema:
  id_cols:
    - map_key
  date_col: clean_date
  target:
    column: y_flag
    good_label: 0
    bad_label: 1
  exclude_cols: []
```

关键配置：

- `id_cols`：记录主键，可由多列组成。
- `date_col`：观察日期，用于生成月份和划分 OOT。
- `target.column`：二分类目标字段。
- `good_label`、`bad_label`：好坏样本标签映射。
- `exclude_cols`：贷后字段、结果字段、人工排除字段及其他疑似泄漏字段。

不要仅凭字段名称自动确认目标含义或数据泄漏风险。

数据文件路径有固定优先级：节点命令中的 `--data` 最高；未提供时读取
`data_contract.yaml` 的 `data.input_path`（相对于契约文件所在目录）；如果契约
没有配置路径，系统只会在 `--project-root` 根目录下自动寻找唯一的 CSV、Parquet
或 Excel 文件。没有文件或存在多个候选文件时会停止并要求用户明确选择，不会
根据文件名猜测数据源。

### 样本诊断与处理配置

编辑 `configs/node_configs/sample-diagnosis.yaml`。诊断阈值只负责产生异常提示；`treatment` 中的动作会写入策略确认文件并绑定文件哈希，样本诊断节点不会执行或保存处理后的数据，后续特征处理节点再按确认策略执行。

当前支持重复主键处理、缺失标签处理、全空特征处理、高缺失样本处理、未完整最新月处理，以及仅作用于 Train 的 LightGBM 类别权重。Test 和 OOT 始终保留自然样本分布，不进行过采样或欠采样。

### 模型配置

编辑 `configs/model_config.yaml`：

```yaml
split:
  oot_months: 2
  test_ratio: 0.2
  random_seed: 42

feature_selection:
  min_iv: 0.02
  max_correlation: 0.9
  max_missing_rate: 0.8
  max_psi: 0.25
  max_unstable_month_ratio: 0.30
  stability_action: review
  correlation_method: spearman

feature_preprocessing:
  numeric:
    invalid_to_null: true
    missing_strategy: native
  categorical:
    strategy: lightgbm_native     # 可改为 drop 使用保守模式
    rare_min_count: 20
    rare_min_rate: 0.001
    unknown_action: missing
    near_unique_rate: 0.98
    max_categories: 100
    high_cardinality_action: drop
  text:
    action: drop
    min_average_length: 64

model:
  objective: binary
  metric: auc
  learning_rate: 0.03
  num_leaves: 31
  num_boost_round: 1000
  early_stopping_rounds: 100
```

自动调参还需配置：

```yaml
training:
  mode: tuning                  # baseline 或 tuning

tuning:
  sampler: tpe
  n_trials: 50
  timeout_seconds: 900
  startup_trials: 10
  cv_folds: 3
  min_bad_samples_per_fold: 20
  auc_gap_penalty: 0.5
  fold_std_penalty: 0.25
  search_space:
    learning_rate: [0.01, 0.08]
    num_leaves: [7, 63]
    max_depth: [3, 8]
```

完整搜索空间见 `configs/model_config.yaml`。这些参数是流程验证阶段的默认值，不代表生产调参标准。

## 使用 Codex Skill

### 安装 Skill

当前 Skill 位于：

```text
model_skills/risk-modeling-pipeline
```

可以将它复制或软链接到个人 Skills 目录：

```bash
mkdir -p "$HOME/.codex/skills"
ln -s \
  "/Users/meixiaohan/Desktop/自动化建模工具 参考/model_skills/risk-modeling-pipeline" \
  "$HOME/.codex/skills/risk-modeling-pipeline"
```

如果目标位置已经存在，请先确认原目录是否需要保留，不要直接覆盖。安装后重新加载 Codex，使 Skill 被发现。

当前 Skill 调用的是 OpenCode 服务端内置的建模引擎；执行时通过 `--project-root` 指向用户工作区，通过 `--engine-root` 指向内置 `model_skills`。

### 对话调用示例

推荐显式指定 Skill 名称：

```text
使用 $risk-modeling-pipeline 检查 data.csv 是否满足建模条件。
```

```text
使用 $risk-modeling-pipeline 分析 data.csv，先识别可能的主键、日期、目标和特征字段，等我确认后再生成 EDA 报告。
```

```text
使用 $risk-modeling-pipeline 准备完整流程。目标字段是 y_flag，坏样本为 1，日期字段是 clean_date，主键是 map_key；先给我看预检、字段建议和实验计划，等我明确确认后再运行。
```

正常情况下，Codex 会先生成 `confirmation_request.json`，向用户汇报关键字段与阻断项并停止；只有用户明确确认后，才会生成批准文件并调用确定性程序。

## 命令行调用

统一入口支持六种模式：

| 模式 | 功能 |
| --- | --- |
| `check` | 检查依赖和核心项目文件 |
| `prepare` | 生成只读预检、AI 建议和待确认请求 |
| `approve` | 记录用户确认并生成哈希绑定的批准文件 |
| `eda` | 执行清理、探索分析并生成 EDA 报告 |
| `model` | 执行切分、特征筛选、LightGBM 训练和模型报告 |
| `all` | 在同一个运行目录中依次执行 EDA 和建模 |

第一步，生成待确认材料：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow \
  --mode prepare \
  --planned-mode all \
  --project-root . \
  --data data.csv \
  --data-contract configs/data_contract.yaml \
  --model-config configs/model_config.yaml \
  --sample-config configs/sample_config.yaml \
  --output-dir outputs/prepare_001 \
  --run-id case_001
```

第二步，用户阅读并明确确认后生成批准文件；若有阻断项，按用户实际确认逐项添加 `--acknowledge-finding 阻断代码`：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow \
  --mode approve \
  --project-root . \
  --confirmation-request outputs/prepare_001/confirmation_request.json \
  --confirmed-by "确认人" \
  --confirm-node data-read \
  --confirm-node sample-diagnosis \
  --acknowledge-finding LATEST_MONTH_INCOMPLETE \
  --approval-file outputs/prepare_001/approval_manifest.json \
  --run-id case_001
```

第三步，执行批准范围内的完整流程：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow \
  --mode all \
  --project-root . \
  --data data.csv \
  --data-contract configs/data_contract.yaml \
  --model-config configs/model_config.yaml \
  --sample-config configs/sample_config.yaml \
  --approval-file outputs/prepare_001/approval_manifest.json \
  --output-dir outputs/run_001 \
  --run-id case_001
```

如果界面只选择部分节点，OpenCode 优先为每个已确认节点分别调用
`python -m workflow <node-id>`。例如读取和样本诊断分别执行：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow data-read \
  --project-root . --engine-root . --data data.csv \
  --data-contract configs/data_contract.yaml \
  --node-config-dir configs/node_configs --output-dir outputs/run_001/data-read

PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow sample-diagnosis \
  --project-root . --engine-root . --data data.csv \
  --data-contract configs/data_contract.yaml \
  --node-config-dir configs/node_configs --output-dir outputs/run_001/sample-diagnosis
```

批量/兼容场景仍可把节点 ID 传给同一个 Skill 的执行器，例如：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow \
  --mode all \
  --steps data-read,sample-diagnosis,feature-processing \
  --project-root . \
  --data data.csv \
  --data-contract configs/data_contract.yaml \
  --sample-config configs/sample_config.yaml \
  --approval-file outputs/prepare_001/approval_manifest.json \
  --output-dir outputs/run_feature_review
```

`--steps` 是批量 runner 的兼容执行范围；不传时才保持 `--mode all` 的完整流程。项目维持一个 `risk-modeling-pipeline` Skill，节点通过独立子命令或批量执行计划运行，不拆成八个彼此难以共享确认状态的 Skill。

只生成 EDA：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow \
  --mode eda \
  --project-root . \
  --data data.csv \
  --data-contract configs/data_contract.yaml \
  --sample-config configs/sample_config.yaml \
  --approval-file outputs/prepare_eda/approval_manifest.json \
  --output-dir outputs/eda_run
```

只训练模型：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow \
  --mode model \
  --project-root . \
  --data data.csv \
  --data-contract configs/data_contract.yaml \
  --model-config configs/model_config.yaml \
  --sample-config configs/sample_config.yaml \
  --approval-file outputs/prepare_model/approval_manifest.json \
  --output-dir outputs/model_run
```

## 输出结果

`eda` 模式主要输出：

- `reports/data_eda_report.xlsx`
- `artifacts/eda/cleaned_data.parquet`
- `artifacts/eda/column_profile.csv`、`column_decisions.csv`
- `artifacts/eda/univariate_overview.csv`、`binning_detail.csv`
- `artifacts/eda/monthly_psi.csv`、`correlation_pairs.csv`
- `prepare_*/sample_diagnostics.json`（确认包）以及 `artifacts/sample_treatment_result.json`

`model` 模式主要输出：

- `reports/model_report.xlsx`
- `models/lightgbm_model.txt`
- `models/lightgbm_model.pkl`（Python Booster）
- `models/model_bundle.pkl`（模型、特征和 Train 拟合预处理方案）
- `models/pmml_export_status.json`；配置 JPMML 转换器后额外生成 `models/lightgbm_model.pmml`
- `artifacts/model/scored_data.parquet`
- `artifacts/model/split_summary.csv`
- `artifacts/model/feature_preprocessing.csv`、`feature_preprocessing.json`
- `artifacts/model/model_matrix.parquet`
- `artifacts/model/feature_selection.csv`
- `artifacts/model/feature_importance.csv`
- `artifacts/model/training_history.csv`
- `candidate_comparison.csv`
- `tuning/optuna_study.db`、`tuning/tuning_trials.csv`、`tuning/cv_fold_metrics.csv`
- `tuning/best_params.json`、`tuning/best_params.yaml`
- `metrics_by_split.csv`
- `lift_detail.csv`
- `monthly_performance.csv`
- `ai_model_review.json`
- 本次运行使用的模型配置和摘要文件

每次正式运行建议使用独立的输出目录，避免覆盖历史结果。

PMML 是可选交付格式。设置 `JPMML_LIGHTGBM_JAR` 环境变量，或在命令中传入
`--pmml-converter /path/to/pmml-lightgbm-example-executable.jar`。没有 Java/JPMML
时训练仍会成功，并在 `pmml_export_status.json` 中记录原因。

## 测试

在 `model_skills` 目录执行：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m unittest discover -s risk-modeling-pipeline/scripts -p "test_*.py"
```

如果当前终端尚未激活环境，可以直接运行：

```bash
conda run -n risk-modeling env PYTHONPATH=risk-modeling-pipeline/scripts python -m unittest discover -s risk-modeling-pipeline/scripts -p "test_*.py"
```

也可以分别运行：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m unittest discover -s risk-modeling-pipeline/scripts/data/test -p "test_*.py"
PYTHONPATH=risk-modeling-pipeline/scripts python -m unittest discover -s risk-modeling-pipeline/scripts/preprocessing/test -p "test_*.py"
PYTHONPATH=risk-modeling-pipeline/scripts python -m unittest discover -s risk-modeling-pipeline/scripts/eda/test -p "test_*.py"
PYTHONPATH=risk-modeling-pipeline/scripts python -m unittest discover -s risk-modeling-pipeline/scripts/modeling/test -p "test_*.py"
```

## 报告解读注意事项

- 同时比较 Train、Test 和 OOT 的 AUC、KS，不只看训练集结果。
- 训练曲线中 Train 与 Test 差距持续扩大时，应重点检查过拟合。
- OOT 坏样本数过少或某个月只有一个目标类别时，月度 AUC、KS 可能不可计算或不稳定。
- PSI 用于识别人群或分数分布漂移，不能单独证明模型效果好坏。
- 高 IV、高重要性字段仍需要检查业务含义、时间穿越和目标泄漏。
- 当前结果适合验证自动化流程；模型上线仍需要业务评审、独立验证、合规检查和监控方案。

## 当前边界与后续方向

当前版本以本地批处理方式运行，并与当前代码仓库绑定。后续可以逐步演进为：

1. 将 `risk-modeling-pipeline/scripts` 封装成可安装的 Python 包。
2. 在现有 LightGBM + Optuna 基础上增加逻辑回归等基准模型与冠军/挑战者管理。
3. 增加更完整的模型版本、实验记录、数据版本和审计信息。
4. 使用 FastAPI 暴露任务提交、状态查询和报告下载接口。
5. 在需要跨系统调用时，再通过 MCP 将建模能力提供给多个智能体或客户端。
