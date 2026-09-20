"""LangGraph stage handlers for the data-read and EDA/diagnosis flow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import polars as pl
import yaml

from core.data_read.contract import contract_from_read_result, load_contract
from core.eda_analysis.metrics import run_eda
from core.feature_screening.screener import load_feature_config, run_feature_screening
from core.modeling.monitor import TrainingMonitor
from core.modeling.lgbm import split_from_manifest
from core.modeling.evaluation import model_quality_components, model_quality_score
from core.modeling.bayesian_tuner import run_bayesian_tuning
from core.modeling.llm_tuner import run_llm_tuning
from core.modeling.model_report import write_model_report
from core.sample_split.splitter import load_split_config, run_sample_split
from nodes.data_read import run_data_read
from .state import ModelingState


@dataclass(frozen=True)
class WorkflowContext:
    """Store immutable paths shared by all nodes in one graph instance."""

    project_root: Path
    skill_root: Path
    data: str | None


def _is_approved(decision: Any) -> bool:
    """将聊天确认值统一解析为布尔批准结果。"""
    if isinstance(decision, dict):
        return bool(decision.get("approved"))
    return bool(decision)


def _cached_frame(state: ModelingState):
    """从 run 级 IPC 缓存恢复数据帧；未启用缓存时返回 None。"""
    cache_path = state.get("data_cache")
    if not cache_path:
        if state.get("cache_data"):
            raise FileNotFoundError("本次运行启用了 cache_data，但找不到数据缓存路径；为避免误读源数据，流程已停止。")
        return None
    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"数据缓存不存在：{path}；为避免误读源数据，流程已停止。")
    # 压缩 Arrow IPC 无法做 memory-map 时会产生误导性的 warning；训练本身不需要映射读取。
    return pl.read_ipc(path, memory_map=False)


class WorkflowStages:
    """Encapsulate node handlers so graph wiring stays declarative."""

    def __init__(self, context: WorkflowContext) -> None:
        """创建绑定到当前项目和 Skill 路径的节点处理器。"""
        self.context = context

    def prepare_data_read(self, state: ModelingState) -> ModelingState:
        """读取并生成 data-read 摘要，然后交给确认节点。"""
        result = run_data_read(
            project_root=self.context.project_root,
            skill_root=self.context.skill_root,
            data=self.context.data,
            run_id=state["run_id"],
            confirm=False,
            return_frame=True,
        )
        data_frame = result.pop("_data_frame", None) if result.get("cache_data", True) else None
        cache_path: str | None = None
        if data_frame is not None:
            cache_file = self.context.project_root / "outputs" / ".langgraph" / f"{state['run_id']}.data-cache.arrow"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            data_frame.write_ipc(cache_file, compression="zstd")
            cache_path = str(cache_file)
        return {"current_node": "data-read", "status": "waiting_confirmation", "pending": result, "data_cache": cache_path, "cache_data": cache_path is not None}

    def wait_data_read(self, state: ModelingState) -> ModelingState:
        """通过 LangGraph interrupt 暂停，等待用户确认字段角色。"""
        from langgraph.types import interrupt

        pending = state["pending"]
        decision = interrupt(
            {
                "type": "node_confirmation",
                "node_id": "data-read",
                "summary": pending["summary"],
                "display_files": pending["display_files"],
                "config": pending["config_path"],
            }
        )
        return {"decision": decision, "status": "resuming"}

    def apply_data_read(self, state: ModelingState) -> ModelingState:
        """应用 data-read 决定；批准后把确认结果传递给 EDA。"""
        if not _is_approved(state.get("decision")):
            return {"status": "cancelled", "current_node": "data-read"}
        pending = state.get("pending") or {}
        data_path = pending.get("data_path") or self.context.data
        approval_dir = self.context.project_root / "configs" / "approvals"
        approval_dir.mkdir(parents=True, exist_ok=True)
        approval_path = approval_dir / "data-read.approval.json"
        approval_path.write_text(json.dumps({"node_id": "data-read", "status": "approved", "config_sha256": pending.get("config_sha256")}, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {**pending, "data_path": data_path, "status": "success", "approval": str(approval_path)}
        return {
            "status": "success",
            "current_node": "data-read",
            "result": result,
            "data_read_result": result,
            "summaries": {"data-read": result["summary"]},
        }

    def run_eda(self, state: ModelingState) -> ModelingState:
        """基于已确认字段角色执行 EDA 和样本诊断。"""
        read_result = state.get("data_read_result") or state.get("result") or state.get("pending") or {}
        fallback = load_contract(self.context.project_root / "configs" / "data_contract.yaml")
        confirmed_contract = contract_from_read_result(read_result, fallback=fallback)
        result = run_eda(
            data_path=read_result.get("data_path") or self.context.data or "",
            output_dir=self.context.project_root / "outputs" / state["run_id"] / "eda-analysis",
            contract=confirmed_contract,
            data_frame=_cached_frame(state),
        )
        return {
            "current_node": "eda-analysis",
            "status": "waiting_confirmation",
            "pending": result,
            "result": result,
            "summaries": {"eda-analysis": result["summary"]},
        }

    def wait_eda(self, state: ModelingState) -> ModelingState:
        """通过 interrupt 暂停，等待用户确认 EDA/样本诊断结果。"""
        from langgraph.types import interrupt

        pending = state["pending"]
        decision = interrupt(
            {
                "type": "node_confirmation",
                "node_id": "eda-analysis",
                "summary": pending["summary"],
                "display_files": pending["display_files"],
            }
        )
        return {"decision": decision, "status": "resuming"}

    def apply_eda(self, state: ModelingState) -> ModelingState:
        """应用 EDA 确认结果并结束当前试验流程。"""
        return {
            "status": "success" if _is_approved(state.get("decision")) else "cancelled",
            "current_node": "eda-analysis",
        }

    def _ensure_config(self, filename: str, asset_name: str) -> Path:
        """按需复制节点配置模板到当前项目。"""
        path = self.context.project_root / "configs" / "node_configs" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        template = self.context.skill_root / "assets" / asset_name
        if not path.exists() and template.is_file():
            shutil.copy2(template, path)
        return path

    def prepare_sample_split(self, state: ModelingState) -> ModelingState:
        """执行样本时间切分并生成待确认摘要。"""
        config_path = self._ensure_config("sample-split.yaml", "sample-split.template.yaml")
        read_result = state.get("data_read_result") or {}
        contract = contract_from_read_result(read_result, fallback=load_contract(self.context.project_root / "configs" / "data_contract.yaml"))
        result = run_sample_split(
            data_path=read_result.get("data_path") or self.context.data or "",
            output_dir=self.context.project_root / "outputs" / state["run_id"] / "sample-split",
            contract=contract,
            data_frame=_cached_frame(state),
            config=load_split_config(config_path),
            config_path=config_path,
        )
        result["config_path"] = str(config_path)
        return {"current_node": "sample-split", "status": "waiting_confirmation", "pending": result, "result": result, "summaries": {"sample-split": result["summary"]}}

    def wait_sample_split(self, state: ModelingState) -> ModelingState:
        """暂停样本切分，等待用户确认时间窗口和分区统计。"""
        from langgraph.types import interrupt

        pending = state["pending"]
        decision = interrupt({"type": "node_confirmation", "node_id": "sample-split", "summary": pending["summary"], "display_files": [pending["summary_markdown"], pending["config_path"]], "config": pending["config_path"]})
        return {"decision": decision, "status": "resuming"}

    def apply_sample_split(self, state: ModelingState) -> ModelingState:
        """应用样本切分确认并进入特征筛选。"""
        if not _is_approved(state.get("decision")):
            return {"status": "cancelled", "current_node": "sample-split"}
        return {"status": "success", "current_node": "sample-split", "result": state.get("pending") or {}}

    def prepare_feature_screening(self, state: ModelingState) -> ModelingState:
        """计算 Train 区分度、跨分区 PSI 和相关性，生成筛选摘要。"""
        config_path = self._ensure_config("feature-screening.yaml", "feature-screening.template.yaml")
        split_result = state.get("result") or state.get("pending") or {}
        read_result = state.get("data_read_result") or {}
        contract = contract_from_read_result(read_result, fallback=load_contract(self.context.project_root / "configs" / "data_contract.yaml"))
        split_dir = Path(split_result.get("manifest", self.context.project_root / "outputs" / state["run_id"] / "sample-split")).parent
        result = run_feature_screening(split_dir=split_dir, data_path=read_result.get("data_path") or self.context.data, source_frame=_cached_frame(state), output_dir=self.context.project_root / "outputs" / state["run_id"] / "feature-screening", contract=contract, config=load_feature_config(config_path), config_path=config_path)
        result["config_path"] = str(config_path)
        split_summary = str(split_result.get("summary", ""))
        result["combined_summary"] = split_summary + "\n\n" + result["summary"]
        result["split_config_path"] = str(split_result.get("config_path", self.context.project_root / "configs" / "node_configs" / "sample-split.yaml"))
        return {"current_node": "feature-screening", "status": "waiting_confirmation", "pending": result, "result": result, "summaries": {"feature-screening": result["summary"]}}

    def wait_feature_screening(self, state: ModelingState) -> ModelingState:
        """暂停特征筛选，等待用户确认保留、复核和删除候选。"""
        from langgraph.types import interrupt

        pending = state["pending"]
        decision = interrupt({"type": "node_confirmation", "node_id": "feature-screening", "summary": pending["summary"], "display_files": [pending.get("summary_markdown", pending["selection"]), pending["selection"], pending.get("split_config_path", pending["config_path"]), pending["config_path"]], "config": pending["config_path"]})
        return {"decision": decision, "status": "resuming"}

    def wait_modeling_data(self, state: ModelingState) -> ModelingState:
        """合并样本切分和特征筛选结果，只进行一次建模数据确认。"""
        from langgraph.types import interrupt

        pending = state["pending"]
        decision = interrupt({"type": "node_confirmation", "node_id": "modeling-data", "summary": pending["combined_summary"], "display_files": [pending.get("summary_markdown", pending["selection"]), pending["selection"], pending["split_config_path"], pending["config_path"]], "config": pending["config_path"]})
        return {"decision": decision, "status": "resuming"}

    def apply_modeling_data(self, state: ModelingState) -> ModelingState:
        """应用一次建模数据确认并把流程推进到模型配置阶段。"""
        if not _is_approved(state.get("decision")):
            return {"status": "cancelled", "current_node": "modeling-data"}
        return {"status": "success", "current_node": "model-config", "result": state.get("pending") or {}, "summaries": {"modeling-data": (state.get("pending") or {}).get("combined_summary", "")}}

    def enter_modeling(self, state: ModelingState) -> ModelingState:
        """标记数据准备完成并进入模型配置节点。"""
        result = dict(state.get("result") or {})
        result["message"] = "建模数据已确认，已进入模型配置阶段。"
        return {"status": "running", "current_node": "model-config", "result": result}

    def prepare_model_config(self, state: ModelingState) -> ModelingState:
        """生成模型配置摘要，并准备贝叶斯/LLM 二选一入口。"""
        config_path = self._ensure_config("model-config.yaml", "model-config.template.yaml")
        data_result = state.get("result") or {}
        selected_columns = data_result.get("selected_columns") or []
        read_result = state.get("data_read_result") or {}
        reserved = set(read_result.get("id_cols") or [])
        reserved.update(name for name in (read_result.get("date_col"), read_result.get("target_col")) if name)
        model_features = [name for name in selected_columns if name not in reserved]
        summary_path = self.context.project_root / "outputs" / state["run_id"] / "model-config" / "model_config_summary.md"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "## 模型配置",
            "",
            "建模数据已确认，请选择一种调参方式。两个调参节点是互斥分支，只会执行你选择的一个。",
            "",
            "| 配置项 | 当前值 | 说明 |",
            "|---|---|---|",
            "| 基础模型 | LightGBM | 当前版本支持二分类 LightGBM |",
            "| 入模字段 | " + str(len(model_features)) + " 列 | 来自已确认的特征筛选结果（不含 ID、日期和 Y） |",
            "| 评估主指标 | Validate KS | 用于调参与模型选择，不使用 OOT 调参 |",
            "| 稳定性评估 | OOT KS / PSI | 只在候选模型评估阶段使用 |",
            "| 调参方式 | 待选择 | `bayesian` 或 `llm`，必须二选一 |",
            "",
            "请选择：`bayesian`（贝叶斯优化）或 `llm`（LLM 调参）。",
        ]
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        pending = {
            "node_id": "model-config",
            "config_path": str(config_path),
            "summary_path": str(summary_path),
            "summary": "\n".join(lines),
            "options": [
                {"id": "bayesian", "label": "贝叶斯优化", "description": "使用 Optuna/TPE 搜索参数"},
                {"id": "llm", "label": "LLM 调参", "description": "根据上一轮指标和参数生成下一组参数"},
            ],
            "selected_columns_count": len(model_features),
        }
        return {"current_node": "model-config", "status": "waiting_confirmation", "pending": pending, "model_config": pending}

    def wait_model_config(self, state: ModelingState) -> ModelingState:
        """暂停模型配置，等待用户明确选择一种调参方式。"""
        from langgraph.types import interrupt

        pending = state["pending"]
        decision = interrupt(
            {
                "type": "model_tuning_selection",
                "node_id": "model-config",
                "summary": pending["summary"],
                "options": pending["options"],
                "display_files": [pending["summary_path"]],
            }
        )
        return {"decision": decision, "status": "resuming"}

    def apply_model_config(self, state: ModelingState) -> ModelingState:
        """校验用户选择，并把流程交给唯一的调参分支。"""
        decision = state.get("decision")
        if not _is_approved(decision):
            return {"status": "cancelled", "current_node": "model-config"}
        method = None
        if isinstance(decision, dict):
            method = decision.get("tuning_method") or decision.get("method")
        method = str(method or "").strip().lower()
        if method not in {"bayesian", "llm"}:
            pending = dict(state.get("pending") or {})
            pending["summary"] = pending.get("summary", "") + "\n\n请选择有效的调参方式：`bayesian` 或 `llm`。"
            return {"status": "waiting_selection", "current_node": "model-config", "pending": pending}
        return {"status": "success", "current_node": "model-config", "tuning_method": method, "model_config": state.get("pending") or {}}

    def route_tuning_method(self, state: ModelingState) -> str:
        """根据确认结果选择唯一调参分支；无效选择回到确认节点。"""
        if state.get("status") == "cancelled":
            return "end"
        method = state.get("tuning_method")
        if method in {"bayesian", "llm"}:
            return method
        return "retry"

    def run_bayesian_optimization(self, state: ModelingState) -> ModelingState:
        """执行 Optuna/TPE LightGBM 搜索，并把最佳模型交给模型审查。"""
        config_path = self._ensure_config("bayesian-optimization.yaml", "bayesian-optimization.template.yaml")
        model_config_path = self._ensure_config("model-config.yaml", "model-config.template.yaml")
        monitor = TrainingMonitor(project_root=self.context.project_root, run_id=state["run_id"], method="bayesian")
        read_result = state.get("data_read_result") or {}
        contract = contract_from_read_result(read_result, fallback=load_contract(self.context.project_root / "configs" / "data_contract.yaml"))
        source = _cached_frame(state)
        manifest_path = self.context.project_root / "outputs" / state["run_id"] / "sample-split" / "split_manifest.json"
        if source is None or not manifest_path.is_file():
            raise FileNotFoundError("贝叶斯优化需要缓存数据和 sample-split 的 split_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets = split_from_manifest(source, contract, manifest)
        previous = state.get("result") or {}
        features = [str(name) for name in (previous.get("selected_columns") or read_result.get("feature_cols") or [])]
        reserved = set(contract.id_cols)
        reserved.update(name for name in (contract.date_col, contract.target_col) if name)
        features = [name for name in features if name not in reserved]
        model_raw = yaml.safe_load(model_config_path.read_text(encoding="utf-8")) or {}
        tuning_raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        model_section = model_raw.get("model", {}) if isinstance(model_raw.get("model", {}), dict) else {}
        model_output_path = self.context.project_root / "outputs" / state["run_id"] / "model" / "model.pkl"
        try:
            result = run_bayesian_tuning(
                train=datasets["train"],
                validate=datasets["validate"],
                oot=datasets["oot"],
                contract=contract,
                features=features,
                baseline_parameters=model_section.get("parameters", {}) if isinstance(model_section.get("parameters", {}), dict) else {},
                settings=dict(tuning_raw),
                monitor=monitor,
                model_output_path=model_output_path,
            )
        except Exception as exc:
            monitor.finish(status="failed")
            result = {"status": "failed", "method": "bayesian", "history": [], "best_metrics": {}, "best_parameters": {}, "error": f"{type(exc).__name__}: {exc}", "model_path": None}
        summary = "## 贝叶斯优化\n\n已完成 LightGBM 基线训练，并使用 Optuna/TPE 逐 trial 搜索参数。目标指标为综合模型质量分：OOT KS 权重最高，同时惩罚 Train/Validate/OOT KS 差距。当前 OOT 参与候选选择；如需将 OOT 完全留作独立最终检验，可在配置中改回 validate_ks。"
        summary += f"\n\n训练监控：`{monitor.events_path}`；共记录 {result.get('completed_trials', 0) + 1} 条迭代。"
        best_trial_label = result.get("best_trial") if result.get("best_trial") is not None else "baseline"
        summary += f"\n\n最佳模型来源：`{best_trial_label}`；最佳模型文件：`{result.get('model_path', '—')}`"
        if result.get("status") == "failed":
            summary += f"\n\n执行失败：{result.get('error')}"
        else:
            summary += "\n\n最佳模型指标：" + json.dumps(result.get("best_metrics", {}), ensure_ascii=False)
        return {"status": "ready_for_model_review", "current_node": "bayesian-optimization", "monitor_dir": str(monitor.output_dir), "iteration_log": str(monitor.events_path), "optimizer_result": {**result, "config_path": str(config_path), "summary": summary, "monitor_dir": str(monitor.output_dir), "iteration_log": str(monitor.events_path)}, "summaries": {"bayesian-optimization": summary}}

    def run_llm_optimization(self, state: ModelingState) -> ModelingState:
        """执行 LightGBM 的 LLM 参数提议、JSON 校验、重训和指标记录闭环。"""
        config_path = self._ensure_config("llm-optimization.yaml", "llm-optimization.template.yaml")
        model_config_path = self._ensure_config("model-config.yaml", "model-config.template.yaml")
        monitor = TrainingMonitor(project_root=self.context.project_root, run_id=state["run_id"], method="llm")
        read_result = state.get("data_read_result") or {}
        contract = contract_from_read_result(read_result, fallback=load_contract(self.context.project_root / "configs" / "data_contract.yaml"))
        source = _cached_frame(state)
        manifest_path = self.context.project_root / "outputs" / state["run_id"] / "sample-split" / "split_manifest.json"
        if source is None or not manifest_path.is_file():
            raise FileNotFoundError("LLM 调参需要缓存数据和 sample-split 的 split_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets = split_from_manifest(source, contract, manifest)
        previous = state.get("result") or {}
        features = [str(name) for name in (previous.get("selected_columns") or read_result.get("feature_cols") or [])]
        reserved = set(contract.id_cols)
        reserved.update(name for name in (contract.date_col, contract.target_col) if name)
        features = [name for name in features if name not in reserved]
        model_raw = yaml.safe_load(model_config_path.read_text(encoding="utf-8")) or {}
        tuning_raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        model_section = model_raw.get("model", {}) if isinstance(model_raw.get("model", {}), dict) else {}
        settings = dict(tuning_raw)
        model_output_path = self.context.project_root / "outputs" / state["run_id"] / "model" / "model.pkl"
        try:
            result = run_llm_tuning(train=datasets["train"], validate=datasets["validate"], oot=datasets["oot"], contract=contract, features=features, baseline_parameters=model_section.get("parameters", {}) if isinstance(model_section.get("parameters", {}), dict) else {}, settings=settings, monitor=monitor, model_output_path=model_output_path)
        except Exception as exc:
            monitor.finish(status="failed")
            result = {"status": "failed", "method": "llm", "history": [], "best_metrics": {}, "best_parameters": {}, "error": f"{type(exc).__name__}: {exc}", "model_path": None}
        summary = "## LLM 调参\n\n已完成 LightGBM 基线训练，并进入 LLM 参数提议、JSON 硬边界校验和候选重训流程。默认按综合模型质量分选择候选：OOT KS 权重最高，并惩罚 Train/Validate/OOT KS 差距。"
        summary += f"\n\n训练监控：`{monitor.events_path}`；共记录 {len(result.get('history', [])) + 1} 条迭代。"
        if result.get("model_path"):
            summary += f"\n\n最佳模型文件：`{result['model_path']}`"
        if result.get("status") == "needs_configuration":
            summary += f"\n\n当前未执行参数提议：{result.get('error')}。请补充 LLM 服务配置后重新运行该分支。"
        elif result.get("status") == "partial":
            summary += f"\n\n部分 LLM 请求未成功：{result.get('error')}。已保留已完成轮次和最佳模型，不会伪造缺失指标；可根据监控日志继续排查或重新运行。"
        if result.get("status") == "failed":
            summary += f"\n\n执行失败：{result.get('error')}"
        if result.get("best_metrics"):
            summary += "\n\n最佳模型指标：" + json.dumps(result["best_metrics"], ensure_ascii=False)
        return {"status": "ready_for_model_review", "current_node": "llm-optimization", "monitor_dir": str(monitor.output_dir), "iteration_log": str(monitor.events_path), "optimizer_result": {"method": "llm", "config_path": str(config_path), "summary": summary, "monitor_dir": str(monitor.output_dir), "iteration_log": str(monitor.events_path), "history": result.get("history", []), "best_parameters": result.get("best_parameters", {}), "best_metrics": result.get("best_metrics", {}), "model_path": result.get("model_path")}, "summaries": {"llm-optimization": summary}}

    def prepare_model_review(self, state: ModelingState) -> ModelingState:
        """统一汇合两个调参分支，并从迭代日志生成模型结果摘要。"""
        optimizer = state.get("optimizer_result") or {}
        summary = optimizer.get("summary", "")
        log_path = state.get("iteration_log") or optimizer.get("iteration_log")
        events: list[dict[str, Any]] = []
        if log_path and Path(log_path).is_file():
            for line in Path(log_path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    events.append(item)
        if events:
            def metric(event: dict[str, Any], dataset: str, name: str = "ks") -> float | None:
                """从模型审查事件中读取一个标量指标，兼容旧版日志。"""
                value = ((event.get("metrics") or {}).get(dataset) or {}).get(name)
                try:
                    return float(value) if value is not None else None
                except (TypeError, ValueError):
                    return None

            scored = [event for event in events if metric(event, "validate") is not None]
            rejected_statuses = {"skipped", "failed", "pruned", "rejected_bounds", "rejected_metric", "rejected_degenerate", "stopped_timeout"}
            selectable = [event for event in scored if str(event.get("status", "complete")) not in rejected_statuses]
            quality_scored = [event for event in (selectable or scored) if model_quality_score(event.get("metrics") or {}) is not None]
            best = max(quality_scored, key=lambda event: model_quality_score(event.get("metrics") or {}) or float("-inf")) if quality_scored else (max(scored, key=lambda event: metric(event, "validate") or float("-inf")) if scored else events[-1])
            best_quality = model_quality_score(best.get("metrics") or {})
            quality = model_quality_components(best.get("metrics") or {})
            train_ks = metric(best, "train")
            validate_ks = metric(best, "validate")
            oot_ks = metric(best, "oot")
            validate_auc = metric(best, "validate", "auc")
            oot_auc = metric(best, "oot", "auc")
            validate_pr_auc = metric(best, "validate", "pr_auc")
            oot_pr_auc = metric(best, "oot", "pr_auc")
            validate_score_psi = metric(best, "validate", "score_psi_vs_train")
            oot_score_psi = metric(best, "oot", "score_psi_vs_train")
            def fmt_metric(value: float | None) -> str:
                return "—" if value is None else f"{value:.5f}"
            summary += "\n\n## 模型结果摘要\n\n"
            summary += "| 项目 | 结果 |\n|---|---:|\n"
            summary += f"| 完成迭代数 | {len(events)} |\n"
            summary += f"| 最佳迭代 | {best.get('iteration', '—')} |\n"
            summary += f"| Train KS | {'—' if train_ks is None else f'{train_ks:.5f}'} |\n"
            summary += f"| Validate KS | {'—' if validate_ks is None else f'{validate_ks:.5f}'} |\n"
            summary += f"| OOT KS | {'—' if oot_ks is None else f'{oot_ks:.5f}'} |\n"
            summary += f"| Validate AUC / Gini | {'—' if validate_auc is None else f'{validate_auc:.5f} / {2 * validate_auc - 1:.5f}'} |\n"
            summary += f"| OOT AUC / Gini | {'—' if oot_auc is None else f'{oot_auc:.5f} / {2 * oot_auc - 1:.5f}'} |\n"
            summary += f"| Validate PR-AUC | {fmt_metric(validate_pr_auc)} |\n"
            summary += f"| OOT PR-AUC | {fmt_metric(oot_pr_auc)} |\n"
            summary += f"| Validate/OOT Score PSI | {fmt_metric(validate_score_psi)} / {fmt_metric(oot_score_psi)} |\n"
            summary += f"| 综合模型质量分 | {fmt_metric(best_quality)} |\n"
            summary += f"| Train-Validate KS gap | {fmt_metric(quality.get('train_validate_gap'))} |\n"
            summary += f"| Validate-OOT KS gap | {fmt_metric(quality.get('validate_oot_gap'))} |\n"
            if optimizer.get("model_path"):
                summary += f"| 最佳模型文件 | `{optimizer['model_path']}` |\n"
            summary += f"\n迭代明细：`{log_path}`"
            result = {"message": summary, "tuning_method": state.get("tuning_method"), "best_iteration": best.get("iteration"), "quality_score": best_quality, "metrics": best.get("metrics", {}), "iterations": len(events), "iteration_log": str(log_path), "model_path": optimizer.get("model_path")}
        else:
            summary += "\n\n## 模型结果摘要\n\n当前尚未记录训练迭代，因此还没有真实的 Train/Validate/OOT 模型指标或模型文件。训练执行器接入后，每轮写入 `iterations.jsonl`，本节点会自动汇总最佳轮次。"
            result = {"message": summary, "tuning_method": state.get("tuning_method"), "iterations": 0, "iteration_log": str(log_path) if log_path else None}
        # 无论调参是否完整，都基于当前已落盘证据生成模型报告；不伪造缺失指标。
        report_path = self.context.project_root / "outputs" / state["run_id"] / "model" / "model_report.html"
        try:
            generated_report = write_model_report(
                iteration_log=log_path or report_path.parent / "iterations.jsonl",
                output_path=report_path,
                run_meta_path=(Path(log_path).with_name("run.json") if log_path else None),
                split_manifest_path=self.context.project_root / "outputs" / state["run_id"] / "sample-split" / "split_manifest.json",
                feature_selection_path=self.context.project_root / "outputs" / state["run_id"] / "feature-screening" / "feature_selection.csv",
                model_path=optimizer.get("model_path"),
            )
            result["report_path"] = str(generated_report)
            summary += f"\n\n模型报告：`{generated_report}`"
        except Exception as exc:
            summary += f"\n\n模型报告生成失败：{type(exc).__name__}: {exc}"
        return {"status": "ready_for_model_review", "current_node": "model-review", "result": result, "summaries": {"model-review": summary}}

    def apply_feature_screening(self, state: ModelingState) -> ModelingState:
        """应用特征筛选确认并结束当前数据准备流程。"""
        return {"status": "success" if _is_approved(state.get("decision")) else "cancelled", "current_node": "feature-screening", "result": state.get("pending") or {}}
