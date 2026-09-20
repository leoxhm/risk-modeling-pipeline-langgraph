"""Independent data-read node; it does not import the legacy engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import polars as pl
import yaml

from core.data_read.contract import DataContract, feature_columns, infer_roles, load_contract
from core.data_read.loader import discover_data_file, load_table


def _config_path(project_root: Path) -> Path:
    """返回当前项目的 data-read 节点配置路径。"""
    return project_root / "configs" / "node_configs" / "data-read.yaml"


def _ensure_config(project_root: Path, skill_root: Path) -> Path:
    """按需从 Skill assets 复制配置模板，并保证项目配置文件存在。"""
    path = _config_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    template = skill_root / "assets" / "data-read.template.yaml"
    if not path.exists() and template.is_file():
        shutil.copy2(template, path)
    if not path.exists():
        path.write_text("node_id: data-read\nparameters: {}\n", encoding="utf-8")
    return path


def _read_parameters(path: Path) -> dict[str, Any]:
    """读取 YAML 中的 parameters 节并规范化为空字典。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(raw.get("parameters", {}) or {})


def _resolve_data(project_root: Path, contract: DataContract, explicit: str | None) -> Path:
    """按显式参数、契约 input_path、工作区唯一文件的优先级解析数据路径。"""
    if explicit:
        path = Path(explicit).expanduser()
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()
    if contract.input_path:
        path = Path(contract.input_path).expanduser()
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()
    return discover_data_file(project_root)


def _summary(data: pl.DataFrame, loaded, contract: DataContract, config_path: Path) -> str:
    """生成供 OpenCode 展示的数据读取 Markdown 摘要表。"""
    features = feature_columns(data, contract)
    feature_text = ", ".join(f"`{name}`" for name in features[:10]) or "—"
    if len(features) > 10:
        feature_text += f" 等（共 {len(features)} 个）"
    id_unique_rate = None
    if contract.id_cols and all(name in data.columns for name in contract.id_cols):
        if len(contract.id_cols) == 1:
            unique_count = data.get_column(contract.id_cols[0]).n_unique()
        else:
            unique_count = data.select(pl.struct(list(contract.id_cols)).n_unique()).item()
        id_unique_rate = unique_count / data.height if data.height else 0.0
    date_parse_rate = None
    if contract.date_col and contract.date_col in data.columns:
        values = data.get_column(contract.date_col).cast(pl.Utf8)
        parsed = None
        for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y%m", "%Y-%m"):
            current = values.str.strptime(pl.Date, format=fmt, strict=False)
            parsed = current if parsed is None else parsed.fill_null(current)
        date_parse_rate = parsed.is_not_null().mean()
    rows = [
        ("数据文件", f"`{loaded.path.name}`"),
        ("文件格式", loaded.source_format.upper()),
        ("样本量", f"{loaded.row_count:,} 行 × {loaded.column_count} 列"),
        ("ID 字段", (", ".join(f"`{name}`" for name in contract.id_cols) or "待确认") + (f"（唯一率 {id_unique_rate:.2%}）" if id_unique_rate is not None else "")),
        ("日期字段", (f"`{contract.date_col}`" if contract.date_col else "待确认") + (f"（解析率 {date_parse_rate:.2%}）" if date_parse_rate is not None else "")),
        ("目标字段（Y）", f"`{contract.target_col}`" if contract.target_col else "待确认"),
        ("好坏标签映射", f"好 = `{contract.good_label}`；坏 = `{contract.bad_label}`"),
        ("特征字段", feature_text),
        ("排除字段", ", ".join(f"`{name}`" for name in contract.exclude_cols) or "—"),
        ("配置来源", f"`{config_path}`"),
    ]
    lines = ["## data-read 结果", "", "| 项目 | 值 |", "|---|---|"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    lines.extend(["", "以上统计由本 Skill 独立使用 Polars 读取并计算。", "请确认字段角色后继续，或直接说明需要修改的字段。"])
    return "\n".join(lines)


def run_data_read(*, project_root: str | Path, skill_root: str | Path, data: str | None, run_id: str, confirm: bool = False, return_frame: bool = False) -> dict[str, Any]:
    """执行数据读取、字段角色推断、摘要落盘及确认/批准产物生成。"""
    root = Path(project_root).expanduser().resolve()
    skill = Path(skill_root).expanduser().resolve()
    config_path = _ensure_config(root, skill)
    parameters = _read_parameters(config_path)
    contract = load_contract(root / "configs" / "data_contract.yaml")
    data_path = _resolve_data(root, contract, data)
    loaded = load_table(data_path, encoding=parameters.get("encoding"), sheet_name=parameters.get("sheet_name", 0), infer_schema_length=int(parameters.get("infer_schema_length", 10_000)))
    contract = infer_roles(loaded.data, contract, parameters)
    features = feature_columns(loaded.data, contract)
    summary = _summary(loaded.data, loaded, contract, config_path)
    output_dir = root / "outputs" / run_id / "data-read"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "data_read_summary.md"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    payload = {"node_id": "data-read", "data_path": str(data_path), "rows": loaded.row_count, "columns": loaded.column_count, "id_cols": list(contract.id_cols), "date_col": contract.date_col, "target_col": contract.target_col, "good_label": contract.good_label, "bad_label": contract.bad_label, "exclude_cols": list(contract.exclude_cols), "feature_cols": features, "config_path": str(config_path), "config_sha256": config_hash, "cache_data": bool(parameters.get("cache_data", True)), "summary": summary}
    (output_dir / "data_read_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not confirm:
        pending = {**payload, "status": "awaiting_user_confirmation", "summary_markdown": str(summary_path)}
        result = {**pending, "display_files": [str(summary_path)]}
        if return_frame:
            result["_data_frame"] = loaded.data
        return result
    approval_dir = root / "configs" / "approvals"
    approval_dir.mkdir(parents=True, exist_ok=True)
    approval_path = approval_dir / "data-read.approval.json"
    approval_path.write_text(json.dumps({"node_id": "data-read", "status": "approved", "config_sha256": config_hash}, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {**payload, "status": "success", "approval": str(approval_path), "display_files": [str(summary_path)]}
    if return_frame:
        result["_data_frame"] = loaded.data
    return result
