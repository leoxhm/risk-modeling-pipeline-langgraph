"""Contract loading and conservative field-role resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml


@dataclass(frozen=True)
class DataContract:
    input_path: str | None = None
    id_cols: tuple[str, ...] = ()
    date_col: str | None = None
    target_col: str | None = None
    good_label: Any = 0
    bad_label: Any = 1
    exclude_cols: tuple[str, ...] = ()
    source: str = "inferred"


def load_contract(path: str | Path) -> DataContract:
    """读取工作区 data_contract.yaml；文件不存在时返回空契约。"""
    contract_path = Path(path).expanduser().resolve()
    if not contract_path.is_file():
        return DataContract()
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    data = raw.get("data", {}) or {}
    schema = raw.get("schema", {}) or {}
    target = schema.get("target", {}) or {}
    return DataContract(
        input_path=data.get("input_path"),
        id_cols=tuple(schema.get("id_cols", ()) or ()),
        date_col=schema.get("date_col"),
        target_col=target.get("column"),
        good_label=target.get("good_label", 0),
        bad_label=target.get("bad_label", 1),
        exclude_cols=tuple(schema.get("exclude_cols", ()) or ()),
        source=str(contract_path),
    )


def _first_existing(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    """按候选名称（忽略大小写）返回第一个实际存在的字段。"""
    lowered = {name.lower(): name for name in columns}
    return next((lowered[name] for name in candidates if name in lowered), None)


def infer_roles(data: pl.DataFrame, contract: DataContract, parameters: dict[str, Any]) -> DataContract:
    """根据节点参数、已有契约和保守候选名推断 ID、日期、标签字段。"""
    columns = set(data.columns)
    id_value = parameters.get("id_col_nm")
    id_cols = (id_value,) if isinstance(id_value, str) and id_value in columns else contract.id_cols
    if not id_cols:
        inferred_id = _first_existing(columns, ("id", "map_key", "user_id", "customer_id", "uid"))
        id_cols = (inferred_id,) if inferred_id else ()
    date_value = parameters.get("dt_col_nm")
    date_col = date_value if isinstance(date_value, str) and date_value in columns else contract.date_col
    if not date_col:
        date_col = _first_existing(columns, ("date", "clean_date", "dt", "event_date", "apply_date"))
    target_value = parameters.get("label_col_nm")
    target_col = target_value if isinstance(target_value, str) and target_value in columns else contract.target_col
    if not target_col:
        target_col = _first_existing(columns, ("y", "label", "target", "y_flag", "bad_flag"))
    return DataContract(
        input_path=contract.input_path,
        id_cols=tuple(item for item in id_cols if item in columns),
        date_col=date_col if date_col in columns else None,
        target_col=target_col if target_col in columns else None,
        good_label=contract.good_label,
        bad_label=contract.bad_label,
        exclude_cols=tuple(item for item in contract.exclude_cols if item in columns),
        source=contract.source,
    )


def feature_columns(data: pl.DataFrame, contract: DataContract) -> list[str]:
    """从数据列中排除 ID、日期、标签及显式排除字段，返回特征列表。"""
    reserved = set(contract.id_cols) | set(contract.exclude_cols)
    reserved.update(item for item in (contract.date_col, contract.target_col) if item)
    return [name for name in data.columns if name not in reserved]


def contract_from_read_result(payload: dict[str, Any], fallback: DataContract | None = None) -> DataContract:
    """从已确认的 data-read 结果恢复 EDA 所需的字段角色和标签映射。"""
    base = fallback or DataContract()
    return DataContract(
        input_path=payload.get("data_path") or base.input_path,
        id_cols=tuple(payload.get("id_cols") or base.id_cols),
        date_col=payload.get("date_col") or base.date_col,
        target_col=payload.get("target_col") or base.target_col,
        good_label=payload.get("good_label", base.good_label),
        bad_label=payload.get("bad_label", base.bad_label),
        exclude_cols=tuple(payload.get("exclude_cols") or base.exclude_cols),
        source=base.source,
    )
