# 自动化建模平台部署指南

本文说明如何在新环境中部署自动化建模项目，并启动 OpenCode 前端供用户使用。

## 0. 打包上传前准备

当前开发目录包含 OpenCode 的本地 `node_modules` 和历史输出，直接压缩会非常大，
并且其中可能包含 macOS 原生依赖。建议拆成“建模引擎包”和“OpenCode 源码包”，
在 Linux 上重新安装 Python/Bun 依赖：

```bash
# 在 model_skills 项目根目录执行
PROJECT_ROOT="$(pwd)"
PACKAGE_DIR="$PROJECT_ROOT/deploy_packages"
mkdir -p "$PACKAGE_DIR"

zip -r "$PACKAGE_DIR/model_skills-engine.zip" . \
  -x 'opencode/*' 'outputs/*' 'data.csv' 'configs/*' \
     'deploy_packages/*' '*/__pycache__/*' '.pytest_cache/*' '.ruff_cache/*' \
     '.git/*' '.DS_Store' '*/.DS_Store'

# 单独打包 OpenCode 源码，不携带本机 node_modules
cd opencode
zip -r "$PACKAGE_DIR/opencode-source.zip" . \
  -x 'node_modules/*' '*/node_modules/*' 'dist/*' '*/dist/*' \
     '.cache/*' '*/.cache/*' '.git/*' '.DS_Store' '*/.DS_Store'

ls -lh "$PACKAGE_DIR"/*.zip
```

在 Linux 服务器上解压为：

```bash
mkdir -p /opt/auto-modeling/model_skills /opt/auto-modeling/opencode
unzip /path/to/model_skills-engine.zip -d /opt/auto-modeling/model_skills
unzip /path/to/opencode-source.zip -d /opt/auto-modeling/opencode
```

压缩包只负责传输源码，不会带 Python 虚拟环境，也不会自动安装依赖；解压后仍需
执行第 4–8 节的系统依赖、Python 环境、Bun 依赖和 Skill 注册步骤。

## 1. 系统架构

平台由三部分组成：

```text
用户浏览器
    │
    ▼
OpenCode Web 前端
    │ HTTP / WebSocket
    ▼
OpenCode 后端服务
    │ 调用 risk-modeling-pipeline Skill
    ▼
OpenCode 内置的 model_skills Python 建模引擎
    │
    ├── 数据读取与清理
    ├── 样本诊断
    ├── 特征预处理与筛选
    ├── LightGBM 训练与 Optuna 调参
    ├── EDA/建模报告
    └── 模型文件与运行产物
```

- `model_skills`：部署在 OpenCode 服务端的内置确定性建模引擎，不属于用户项目。
- `risk-modeling-pipeline`：内置 Skill 适配层，负责编排对话、确认门禁和引擎调用。
- 用户项目：只保存数据文件、配置文件和运行输出。
- `opencode`：AI 对话、节点流程和 Web 前端。

## 2. 推荐目录结构

```text
/opt/auto-modeling/
├── model_skills/
│   ├── .venv/
│   ├── requirements.txt
│   └── risk-modeling-pipeline/
│       └── scripts/                 # workflow entry point and engine
├── workspaces/
│   └── user-001/
│       ├── data.csv
│       ├── configs/
│       └── outputs/
└── opencode/
```

## 3. 依赖清单与内网风险

### Python 运行时

推荐 Linux x86_64、Python 3.11。项目直接依赖：

| 包 | 用途 | 内网部署注意事项 |
|---|---|---|
| `polars` | CSV/Parquet 数据读取和计算 | 有原生 wheel，需匹配 Python/CPU 架构 |
| `fastexcel` | Excel 文件读取 | 只读 CSV/Parquet 时仍建议安装，避免 Loader 能力不完整 |
| `PyYAML` | YAML 配置读取 | 纯 Python/轻量依赖 |
| `numpy`、`scipy` | LightGBM、Toad 和指标计算 | 二进制包，必须提前准备对应 Linux wheel |
| `lightgbm` | 模型训练和预测 | 可能依赖 `libgomp1`；老版本 glibc 或 ARM 环境要单独验证 |
| `optuna` | 自动调参 | 会带入 `SQLAlchemy`、`Alembic`、`colorlog`、`tqdm` 等间接依赖 |
| `toad` | IV、KS、PSI | 当前默认指标后端，不能只安装 LightGBM |
| `XlsxWriter` | EDA/模型 Excel 报告 | 纯 Python，报告生成不依赖 Node 或 `@oai/artifact-tool` |

`requirements.txt` 已补充 `numpy` 和 `scipy`，并且环境检查会验证
`polars/fastexcel/yaml/numpy/scipy/lightgbm/optuna/toad/xlsxwriter` 是否存在。
Toad 的包版本或 Linux 架构不匹配，是内网部署中最容易遗漏的部分。

### OpenCode 运行时

- Bun：项目锁定 `bun@1.3.14`，用于安装和运行 OpenCode。
- Node.js：建议准备 Node.js 22/24，主要用于部分前端构建工具和原生依赖兜底。
- OpenCode 前端依赖中包含 GitHub 和 `pkg.pr.new` 来源；完全断网时不能直接执行首次 `bun install`，应在有网的同架构机器预装依赖后整体迁移，或配置内网 npm/Bun 镜像。
- `node-pty`、`tree-sitter`、`esbuild` 等依赖可能触发原生构建，准备 `gcc/g++/make` 更稳妥。

### 可选 PMML 运行时

只有需要 PMML 时才安装 Java 11+ 和 JPMML-LightGBM 可执行 JAR；否则不影响
Python 模型训练、pickle 模型和 Excel 报告生成。

## 4. 安装系统依赖

以 Ubuntu 为例：

```bash
sudo apt update
sudo apt install -y git ca-certificates python3.11 python3.11-venv \
  python3.11-dev libgomp1 libstdc++6 build-essential
```

安装 Bun（联网环境）：

```bash
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc
```

## 5. 安装 Python 建模环境（联网安装）

联网环境可以直接执行本节；完全内网环境请跳到第 6 节使用 wheelhouse，二选一即可。

```bash
cd /opt/auto-modeling/model_skills

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

依赖包括：

- Polars
- LightGBM
- Optuna
- Toad
- XlsxWriter
- PyYAML

执行环境检查：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow \
  --mode check \
  --engine-root /opt/auto-modeling/model_skills
```

如果服务器统一使用 Conda，也可以使用同一套依赖文件，不需要额外创建 `.venv`：

```bash
conda create -n auto-modeling python=3.11 -y
conda activate auto-modeling
cd /opt/auto-modeling/model_skills
python -m pip install --upgrade pip
python -m pip install -r requirements-full.txt
```

`requirements-full.txt` 在运行时依赖之外增加了 `ipykernel`、`pytest` 和 `ruff`，
适合从 JupyterLab 部署和做环境验证；只运行引擎时使用 `requirements.txt` 即可。
使用 Conda 时，启动 OpenCode 后端前必须先 `conda activate auto-modeling`，或将
`$CONDA_PREFIX/bin` 放到 `PATH` 的最前面，确保 Skill 使用同一个 Python。

## 6. 内网离线安装 Python 依赖

不要在内网机器上直接访问 PyPI。先在一台与目标 Linux 的 Python 版本、CPU
架构和 glibc 兼容的有网机器准备 wheelhouse：

```bash
python3.11 -m venv build-venv
source build-venv/bin/activate
python -m pip install --upgrade pip
python -m pip download -r requirements.txt -d wheelhouse
```

将 `wheelhouse/`、项目代码和 requirements 文件复制到内网服务器，然后：

```bash
source /opt/auto-modeling/model_skills/.venv/bin/activate
python -m pip install --no-index --find-links /opt/auto-modeling/wheelhouse \
  -r /opt/auto-modeling/model_skills/requirements.txt
```

Conda 离线环境可将 `requirements-full.txt` 替换到上面的 `-r` 参数；wheelhouse
必须在相同 Linux 架构和 Python 版本上准备，不能直接复用 macOS wheel。

安装后先执行：

```bash
PYTHONPATH=risk-modeling-pipeline/scripts python -m workflow \
  --mode check \
  --engine-root /opt/auto-modeling/model_skills
```

只有 `ready: true` 才开始 OpenCode 联调。若 `pip download` 找不到目标平台
的 wheel，应在同平台准备 wheel；不要在 macOS 上下载后直接复制给 Linux。

## 7. 安装 OpenCode 依赖

```bash
cd /opt/auto-modeling/opencode
bun install --frozen-lockfile
```

### 7.1 内网安装 OpenCode 依赖

如果服务器可以访问公司内部 npm/Bun 镜像，优先配置镜像后再安装：

```bash
cd /opt/auto-modeling/opencode
bun install --frozen-lockfile \
  --registry "https://<your-internal-registry>/"
```

也可以在用户级 `~/.bunfig.toml` 或项目 `bunfig.toml` 中配置
`[install].registry`。认证信息请通过内网 `.npmrc`、Bun 配置或环境变量提供，
不要写进命令历史。当前锁文件还包含 `ghostty-web` 的 GitHub 依赖和
`@solidjs/start` 的 `pkg.pr.new` 依赖；内部镜像必须能代理这些来源，否则使用下方
的离线缓存方案。

完全断网时，在同架构 Linux 机器（x86_64 或 aarch64）上使用相同 Bun 版本准备：

```bash
mkdir -p /tmp/opencode-offline/cache
cd /path/to/opencode
bun install --frozen-lockfile \
  --cache-dir /tmp/opencode-offline/cache
tar -czf /tmp/opencode-node-modules.tgz -C /path/to/opencode node_modules
tar -czf /tmp/opencode-bun-cache.tgz -C /tmp/opencode-offline cache
```

将两个压缩包复制到内网服务器后：

```bash
cd /opt/auto-modeling/opencode
tar -xzf /path/to/opencode-node-modules.tgz
mkdir -p offline-cache
tar -xzf /path/to/opencode-bun-cache.tgz \
  -C offline-cache --strip-components=1
bun install --offline --frozen-lockfile \
  --cache-dir /opt/auto-modeling/opencode/offline-cache
```

缓存和 `node_modules` 必须来自相同 Linux 架构；不要使用 macOS 依赖。`--offline`
要求所需包、Git 依赖和 tarball 都已经在缓存中，否则安装会失败。

## 8. 注册自动建模 Skill

将 `model_skills` 注册为 OpenCode 的内置 Skill 路径，不要把 Skill 复制到用户项目中：

编辑全局 OpenCode 配置 `~/.config/opencode/opencode.json`：

注册后目录结构如下：

```text
model_skills/
├── risk-modeling-pipeline/
│   ├── SKILL.md
│   ├── scripts/                    # workflow + merged Python engine
│   ├── assets/
│   └── references/
└── .venv/
```

也可以在 OpenCode 配置中指定 Skill 路径：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "skills": {
    "paths": [
      "/opt/auto-modeling/model_skills"
    ]
  }
}
```

## 9. 本地开发启动

### 9.1 启动 OpenCode 后端

```bash
cd /opt/auto-modeling/opencode

export OPENCODE_SERVER_PASSWORD="请替换为强密码"
export PATH="/opt/auto-modeling/model_skills/.venv/bin:$PATH"

bun run dev -- serve \
  --hostname 0.0.0.0 \
  --port 4096 \
  --cors http://服务器IP:3000
```

### 9.2 启动 Web 前端

另开一个终端：

```bash
cd /opt/auto-modeling/opencode/packages/app

export VITE_OPENCODE_SERVER_HOST="服务器IP"
export VITE_OPENCODE_SERVER_PORT="4096"

bun run dev -- \
  --host 0.0.0.0 \
  --port 3000
```

浏览器访问：

```text
http://服务器IP:3000
```

如果前后端跨域，需要在后端增加 CORS：

```bash
bun run dev -- serve \
  --hostname 0.0.0.0 \
  --port 4096 \
  --cors http://服务器IP:3000
```

### 9.3 从 JupyterLab 启动

JupyterLab 只作为执行终端，不建议在 Notebook 单元中以前台方式运行服务，否则
单元会一直占用 Kernel。使用 `nohup` 后台启动，并将日志和 PID 写入固定目录。

后端单元：

```bash
%%bash
set -euo pipefail
ROOT=/opt/auto-modeling
BUN_BIN="$(command -v bun || true)"
SERVER_IP="服务器IP"
test -n "$BUN_BIN" || { echo "bun not found in PATH" >&2; exit 1; }
export PATH="$ROOT/model_skills/.venv/bin:$PATH"
mkdir -p "$ROOT/logs" "$ROOT/run"
export OPENCODE_SERVER_PASSWORD='请替换为强密码'
nohup "$BUN_BIN" --cwd "$ROOT/opencode" run dev -- serve \
  --hostname 0.0.0.0 \
  --port 4096 \
  --cors "http://${SERVER_IP}:3000" \
  > "$ROOT/logs/opencode-backend.log" 2>&1 &
echo $! > "$ROOT/run/opencode-backend.pid"
echo "backend pid=$(cat "$ROOT/run/opencode-backend.pid")"
```

前端单元：

```bash
%%bash
set -euo pipefail
ROOT=/opt/auto-modeling
BUN_BIN="$(command -v bun || true)"
SERVER_IP="服务器IP"
test -n "$BUN_BIN" || { echo "bun not found in PATH" >&2; exit 1; }
mkdir -p "$ROOT/logs" "$ROOT/run"
export VITE_OPENCODE_SERVER_HOST="$SERVER_IP"
export VITE_OPENCODE_SERVER_PORT=4096
nohup "$BUN_BIN" --cwd "$ROOT/opencode/packages/app" run dev -- \
  --host 0.0.0.0 \
  --port 3000 \
  > "$ROOT/logs/opencode-frontend.log" 2>&1 &
echo $! > "$ROOT/run/opencode-frontend.pid"
echo "frontend pid=$(cat "$ROOT/run/opencode-frontend.pid")"
```

检查服务：

```bash
!curl -fsS http://服务器IP:4096/api/health
!curl -I http://服务器IP:3000
```

停止服务：

```bash
!kill "$(cat /opt/auto-modeling/run/opencode-frontend.pid)"
!kill "$(cat /opt/auto-modeling/run/opencode-backend.pid)"
```

如果 JupyterLab 运行在另一台机器，浏览器必须能直接访问服务器的 `3000` 和
`4096` 端口；JupyterLab 本身的访问地址不会自动代理这两个端口。

## 10. 内网长期运行

内部使用不需要 Nginx 或反向代理，直接运行两个服务即可：

- 前端：`3000`
- 后端 API：`4096`

### 10.1 方式一：直接运行开发前端

这种方式最适合内部测试和频繁修改页面的场景。

后端保持运行：

```bash
cd /opt/auto-modeling/opencode

export OPENCODE_SERVER_PASSWORD="请替换为强密码"
export PATH="/opt/auto-modeling/model_skills/.venv/bin:$PATH"

bun run dev -- serve \
  --hostname 0.0.0.0 \
  --port 4096 \
  --cors http://服务器IP:3000
```

前端另开终端：

```bash
cd /opt/auto-modeling/opencode/packages/app

export VITE_OPENCODE_SERVER_HOST="服务器IP"
export VITE_OPENCODE_SERVER_PORT="4096"

bun run dev -- \
  --host 0.0.0.0 \
  --port 3000
```

用户访问：

```text
http://服务器IP:3000
```

### 10.2 方式二：构建前端后使用 Vite Preview

```bash
cd /opt/auto-modeling/opencode
bun run --cwd packages/app build
```

构建产物：

```text
/opt/auto-modeling/opencode/packages/app/dist
```

启动后端：

```bash
cd /opt/auto-modeling/opencode

export OPENCODE_SERVER_PASSWORD="请替换为强密码"

bun run dev -- serve \
  --hostname 0.0.0.0 \
  --port 4096 \
  --cors http://服务器IP:3000
```

启动构建后的前端：

```bash
cd /opt/auto-modeling/opencode/packages/app
bun run serve -- --host 0.0.0.0 --port 3000
```

打开前端后，在 OpenCode 的服务器设置中填写：

```text
服务器地址：http://服务器IP:4096
用户名：opencode
密码：OPENCODE_SERVER_PASSWORD 的值
```

如果不想每次手工填写服务器地址，建议使用“方式一”，因为开发前端可以通过
`VITE_OPENCODE_SERVER_HOST` 和 `VITE_OPENCODE_SERVER_PORT` 默认连接后端。

## 11. 用户使用流程

1. 用户打开 Web 页面。
2. 进入“AI 建模”模式。
3. 选择自动建模节点。
4. 上传数据文件或指定项目数据路径。
5. AI 调用 `risk-modeling-pipeline` Skill。
6. 系统依次执行：
   - 读取数据与契约校验
   - 样本诊断
   - 用户确认样本处理方案
   - 特征预处理与筛选
   - 用户确认特征方案
   - 生成模型配置
   - 用户确认模型配置
   - LightGBM 训练与 Optuna 调参
   - 用户确认训练结果
   - 模型审查
   - 生成 EDA 报告、建模报告和模型文件

## 12. 运行产物

每次运行都会生成独立输出目录，主要包括：

- `data_eda_report.xlsx`
- `model_report.xlsx`
- `model_summary.json`
- `lightgbm_model.pkl`
- `model_bundle.pkl`
- `metrics_by_split.csv`
- `lift_detail.csv`
- `feature_importance.csv`
- `scored_data.parquet`
- `lightgbm_model.txt`
- `pmml_export_status.json`；配置 JPMML 转换器后额外生成 `lightgbm_model.pmml`
- `run_events.jsonl`
- `run_state.json`

## 13. 安全注意事项

- 内网环境仍建议设置 `OPENCODE_SERVER_PASSWORD`。
- 防火墙只允许内网网段访问 3000 和 4096 端口。
- 不要把 3000、4096 端口暴露到公网。
- 每个用户建议使用独立工作目录。
- OpenCode 后端运行账号必须能读取/执行 `model_skills`，并对用户工作区的
  `configs/` 和 `outputs/` 具有读写权限。
- OpenCode 后端进程的 `PATH` 必须优先包含
  `/opt/auto-modeling/model_skills/.venv/bin`，确保 Skill 调用的是建模环境的
  Python，而不是 JupyterLab 或系统 Python。
- 风控数据属于敏感数据，应避免上传到不受控的外部环境。
- 不要把 `data.csv`、API Key 或密码提交到 Git 仓库。
- 生产模型上线前仍需人工完成业务、合规和独立验证。

### 13.1 模型文件格式

训练默认保存 LightGBM 原生文本模型、Python pickle 和包含预处理元数据的
`model_bundle.pkl`。PMML 不是 Python 核心依赖；如需生成 PMML，应安装 Java 11
和 JPMML-LightGBM 可执行 JAR，然后设置：

```bash
export JPMML_LIGHTGBM_JAR=/opt/auto-modeling/tools/pmml-lightgbm-example-executable.jar
```

也可以在运行命令中传入 `--pmml-converter`。转换失败不会删除已生成的文本或
pickle 模型，具体状态记录在 `pmml_export_status.json`。

## 14. 常见问题

### 前端无法连接后端

检查：

```bash
curl http://服务器IP:4096/api/health
```

确认后端端口、防火墙、CORS 和前端的 `VITE_OPENCODE_SERVER_HOST` 配置。

### Skill 没有被发现

确认内置 Skill 文件存在：

```text
/opt/auto-modeling/model_skills/risk-modeling-pipeline/SKILL.md
```

并确认 `~/.config/opencode/opencode.json` 的 `skills.paths` 包含
`/opt/auto-modeling/model_skills`，然后重新启动 OpenCode。Skill 配置通常在进程启动时加载。

### 模型训练失败

先重新检查 Python 环境：

```bash
source /opt/auto-modeling/model_skills/.venv/bin/activate
PYTHONPATH=/opt/auto-modeling/model_skills/risk-modeling-pipeline/scripts \
python -m workflow \
  --mode check \
  --engine-root /opt/auto-modeling/model_skills
```

重点确认 `lightgbm`、`optuna`、`toad` 和 `polars` 是否安装成功。
