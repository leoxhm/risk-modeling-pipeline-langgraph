# 独立训练监控网页

这是一个与 OpenCode、LangGraph 执行图和建模流程完全解耦的本地查看器。它只读取用户选择的 `iterations.jsonl` 文件，不会启动或修改任何训练任务。

## 启动

无需安装前端依赖：

```bash
python monitor_web/serve.py
```

打开：<http://127.0.0.1:8765/index.html>

然后选择：

```text
outputs/<run_id>/modeling-monitor/iterations.jsonl
```

## 展示内容

- 当前轮次、最佳综合质量分、对应 OOT KS、接受轮次；
- 全部 baseline/trial/LLM 轮次的 Train、Validate、OOT KS/AUC；
- 点击任意轮次查看完整参数和判定原因；
- 读取 `fit_history` 绘制 Train/Validate 拟合曲线；
- 每秒重新读取文件，支持训练过程中实时追加 JSONL；
- 使用浏览器 File System Access API 时，选择的文件会持续读取最新内容；不支持该 API 的浏览器可使用普通文件选择作为静态查看。

页面不依赖 OpenCode SDK、LangGraph Server、Node.js、Bun 或第三方图表库。
