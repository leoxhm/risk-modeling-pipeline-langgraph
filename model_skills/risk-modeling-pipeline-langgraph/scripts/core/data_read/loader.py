"""Polars-based, self-contained tabular data loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl


SupportedFormat = Literal["csv", "parquet", "excel"]
SUFFIXES: dict[str, SupportedFormat] = {".csv": "csv", ".parquet": "parquet", ".xlsx": "excel", ".xls": "excel"}


@dataclass(frozen=True)
class LoadedTable:
    data: pl.DataFrame
    path: Path
    source_format: SupportedFormat

    @property
    def row_count(self) -> int:
        """返回已加载表格的行数。"""
        return self.data.height

    @property
    def column_count(self) -> int:
        """返回已加载表格的列数。"""
        return self.data.width


def discover_data_file(project_root: str | Path) -> Path:
    """在工作区根目录发现唯一数据文件；多个候选文件时要求明确指定。"""
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"工作区不存在：{root}")
    candidates = sorted(item for item in root.iterdir() if item.is_file() and not item.name.startswith(".") and item.suffix.lower() in SUFFIXES)
    if not candidates:
        raise FileNotFoundError("工作区根目录没有找到 CSV、Parquet 或 Excel 数据文件")
    if len(candidates) > 1:
        raise ValueError("发现多个数据文件（" + ", ".join(item.name for item in candidates) + "），请通过 --data 明确指定")
    return candidates[0]


def load_table(path: str | Path, *, encoding: str | None = None, sheet_name: str | int = 0, infer_schema_length: int = 10_000) -> LoadedTable:
    """使用 Polars 读取 CSV、Parquet 或 Excel，并返回带元信息的表对象。"""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"数据文件不存在：{source}")
    source_format = SUFFIXES.get(source.suffix.lower())
    if source_format is None:
        raise ValueError(f"不支持的数据格式：{source.suffix}")
    if source_format == "csv":
        options: dict[str, Any] = {"infer_schema_length": infer_schema_length}
        if encoding:
            options["encoding"] = encoding
        frame = pl.read_csv(source, **options)
    elif source_format == "parquet":
        frame = pl.read_parquet(source)
    elif isinstance(sheet_name, str):
        frame = pl.read_excel(source, sheet_name=sheet_name)
    else:
        frame = pl.read_excel(source, sheet_id=int(sheet_name) + 1)
    return LoadedTable(frame, source, source_format)


def scan_table(path: str | Path, *, encoding: str | None = None, infer_schema_length: int = 10_000) -> pl.LazyFrame:
    """返回适合大文件处理的 LazyFrame；Excel 因不支持惰性读取而一次性物化。"""
    source = Path(path).expanduser().resolve()
    source_format = SUFFIXES.get(source.suffix.lower())
    if source_format == "csv":
        options: dict[str, Any] = {"infer_schema_length": infer_schema_length}
        if encoding:
            options["encoding"] = encoding
        return pl.scan_csv(source, **options)
    if source_format == "parquet":
        return pl.scan_parquet(source)
    # Excel has no lazy reader in Polars; callers should convert it to Parquet.
    return load_table(source).data.lazy()
