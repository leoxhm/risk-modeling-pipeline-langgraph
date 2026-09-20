"""时间切分和建模样本产物生成。"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import json
from pathlib import Path
import statistics
from typing import Any

import polars as pl
import yaml

from core.data_read.contract import DataContract, feature_columns
from core.data_read.loader import load_table


def _month_key(value: Any) -> str | None:
    """将日期、字符串或数值日期统一为 YYYYMM 月份键。"""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y%m")
    text = str(value).strip()
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) >= 6:
        return digits[:6]
    return None


def load_split_config(path: str | Path) -> dict[str, Any]:
    """读取精简的 modeling_split.yaml 配置。"""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return dict(raw)


def _dataset_summary(frame: pl.DataFrame, contract: DataContract, months: list[str]) -> dict[str, Any]:
    """生成单个 Train/Validate/OOT 分区的样本统计。"""
    bad_count = 0
    if contract.target_col and contract.target_col in frame.columns:
        bad_count = int(frame.select((pl.col(contract.target_col) == contract.bad_label).sum()).item() or 0)
    return {
        "sample_count": frame.height,
        "bad_count": bad_count,
        "good_count": frame.height - bad_count,
        "bad_rate": bad_count / frame.height if frame.height else 0.0,
        "month_start": min(months) if months else None,
        "month_end": max(months) if months else None,
        "months": sorted(set(months)),
    }


def _split_months(month_counts: Counter[str], config: dict[str, Any]) -> tuple[set[str], set[str], set[str], list[str]]:
    """根据月份计数确定排除、Train、Validate 和 OOT 月份集合。"""
    ordered = sorted(month_counts)
    if not ordered:
        raise ValueError("日期字段没有可解析的月份，无法使用时间切分")
    excluded: list[str] = []
    ratio = float(config.get("incomplete_month_ratio", 0.30))
    if bool(config.get("exclude_latest_incomplete", True)) and len(ordered) >= 2:
        typical = statistics.median(month_counts.values())
        latest = ordered[-1]
        if month_counts[latest] < typical * ratio:
            excluded.append(latest)
            ordered = ordered[:-1]
    oot_months = max(0, int(config.get("oot_months", 2)))
    if len(ordered) <= oot_months:
        raise ValueError("可用于建模的完整月份少于 OOT 月份数，请减少 oot_months 或扩大数据时间范围")
    oot = set(ordered[-oot_months:]) if oot_months else set()
    history = [month for month in ordered if month not in oot]
    validate_ratio = min(max(float(config.get("validate_ratio", 0.20)), 0.0), 0.80)
    target_validate = max(1, int(sum(month_counts[month] for month in history) * validate_ratio)) if history else 0
    validate: set[str] = set()
    accumulated = 0
    for month in reversed(history):
        if accumulated >= target_validate and validate:
            break
        validate.add(month)
        accumulated += month_counts[month]
    train = set(history) - validate
    if not train or not validate:
        raise ValueError("Train 或 Validate 为空，请调整 validate_ratio 或时间范围")
    return train, validate, oot, excluded


def run_sample_split(*, data_path: str | Path, output_dir: str | Path, contract: DataContract, data_frame: pl.DataFrame | None = None, config: dict[str, Any] | None = None, config_path: str | Path | None = None) -> dict[str, Any]:
    """执行时间切分，保存轻量清单和摘要，不生成分区 Parquet。"""
    settings = {"strategy": "time", "validate_ratio": 0.20, "oot_months": 2, "exclude_latest_incomplete": True, "incomplete_month_ratio": 0.30}
    settings.update(config or {})
    if settings.get("strategy", "time") != "time":
        raise ValueError("当前 sample-split 节点只支持 strategy: time")
    if not contract.date_col:
        raise ValueError("data-read 契约未确认日期字段，无法进行时间切分")
    frame = data_frame if data_frame is not None else load_table(data_path).data
    if contract.date_col not in frame.columns:
        raise ValueError(f"日期字段不存在：{contract.date_col}")
    months = [_month_key(value) for value in frame.get_column(contract.date_col).to_list()]
    valid_months = [month for month in months if month is not None]
    month_counts = Counter(valid_months)
    train_months, validate_months, oot_months, excluded_months = _split_months(month_counts, settings)
    split_labels = [
        "train" if month in train_months else "validate" if month in validate_months else "oot" if month in oot_months else "excluded"
        for month in months
    ]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_frame = frame.with_columns(pl.Series("_split", split_labels))
    datasets: dict[str, pl.DataFrame] = {name: split_frame.filter(pl.col("_split") == name).drop("_split") for name in ("train", "validate", "oot", "excluded")}
    summaries = {
        name: _dataset_summary(dataset, contract, [month for month, label in zip(months, split_labels) if label == name and month is not None])
        for name, dataset in datasets.items()
    }
    manifest = {
        "node_id": "sample-split",
        "data_path": str(Path(data_path).expanduser().resolve()),
        "strategy": "time",
        "date_col": contract.date_col,
        "target_col": contract.target_col,
        "config_path": str(config_path) if config_path else None,
        "train_months": sorted(train_months),
        "validate_months": sorted(validate_months),
        "oot_months": sorted(oot_months),
        "excluded_months": excluded_months,
        "invalid_date_rows": len(months) - len(valid_months),
        "datasets": {name: summaries[name] for name in ("train", "validate", "oot")},
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["## 样本切分结果", "", "| 数据集 | 时间范围 | 月份数 | 样本数 | 坏样本数 | 坏账率 | 用途 |", "|---|---|---:|---:|---:|---:|---|"]
    purposes = {"train": "特征筛选与模型训练", "validate": "调参与模型选择", "oot": "最终稳定性评估", "excluded": "不进入建模"}
    for name in ("train", "validate", "oot", "excluded"):
        item = summaries[name]
        if item["sample_count"]:
            lines.append(f"| {name.title()} | {item['month_start']}–{item['month_end']} | {len(item['months'])} | {item['sample_count']:,} | {item['bad_count']:,} | {item['bad_rate']:.5%} | {purposes[name]} |")
    if excluded_months:
        lines.extend(["", f"自动排除不完整月份：{', '.join(excluded_months)}（样本量低于历史月份中位数的 {float(settings['incomplete_month_ratio']):.0%}）。原始数据未删除。"])
    if manifest["invalid_date_rows"]:
        lines.extend(["", f"警告：有 {manifest['invalid_date_rows']} 行日期无法解析，未进入建模分区。"])
    return {"status": "success", "node_id": "sample-split", "summary": "\n".join(lines), "manifest": str(output / "split_manifest.json"), "train_months": sorted(train_months), "validate_months": sorted(validate_months), "oot_months": sorted(oot_months), "excluded_months": excluded_months, "artifacts": [str(output / "split_manifest.json")]}
