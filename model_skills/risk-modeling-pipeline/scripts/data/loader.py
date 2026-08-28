"""Load tabular source files without applying modeling transformations."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from logger import get_logger

SupportedFormat = Literal["csv", "excel", "parquet"]
logger = get_logger(__name__)

class DataLoaderError(ValueError):
    """Raised when a source file cannot be loaded as a supported table."""


@dataclass(frozen=True)
class DataLoadResult:
    """Raw table and immutable metadata collected at load time."""

    data: pl.DataFrame
    path: Path
    source_format: SupportedFormat
    row_count: int
    column_count: int


_SUFFIX_TO_FORMAT: dict[str, SupportedFormat] = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
}


def discover_data_file(project_root: str | Path) -> Path:
    """Find the single supported tabular file at a workspace root.

    Discovery is intentionally conservative. It only inspects files directly
    under ``project_root`` and never guesses between multiple datasets. Callers
    should prefer an explicit ``--data`` path or the contract's
    ``data.input_path`` whenever either is available.
    """
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {root}")
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in _SUFFIX_TO_FORMAT
    )
    if not candidates:
        allowed = ", ".join(sorted(_SUFFIX_TO_FORMAT))
        raise DataLoaderError(
            f"No data file found directly under project root {root}. "
            f"Supported extensions: {allowed}. Provide --data or set "
            "data.input_path in data_contract.yaml."
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise DataLoaderError(
            f"Multiple data files found under project root {root}: {names}. "
            "Provide an explicit --data path so the source is not guessed."
        )
    logger.info("Discovered workspace data file: %s", candidates[0])
    return candidates[0]


def _detect_format(path: Path) -> SupportedFormat:
    try:
        return _SUFFIX_TO_FORMAT[path.suffix.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(_SUFFIX_TO_FORMAT))
        raise DataLoaderError(
            f"Unsupported input format: '{path.suffix or '<no extension>'}'. "
            f"Supported extensions: {allowed}."
        ) from exc


def load_table(
    path: str | Path,
    *,
    encoding: str | None = None,
    sheet_name: str | int = 0,
    **read_options: Any,
) -> DataLoadResult:
    """Load one CSV, Excel sheet, or Parquet file as a raw Polars DataFrame.

    Keep this function limited to I/O. Date parsing, deduplication, missing-value
    handling, and target validation belong to later pipeline stages.

    Args:
        path: Source file path. Supported extensions are csv, xls, xlsx, parquet.
        encoding: Optional CSV encoding, for example ``"utf-8-sig"`` or ``"gbk"``.
        sheet_name: Excel sheet name or zero-based index. Ignored for CSV and Parquet.
        **read_options: Extra keyword arguments passed to the matching Polars reader.

    Returns:
        The raw DataFrame plus source path, format, and shape metadata.

    Raises:
        FileNotFoundError: The input path does not exist.
        DataLoaderError: The file suffix is unsupported or Polars cannot read it.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        logger.error("Input file does not exist: %s", source_path)
        raise FileNotFoundError(f"Input file does not exist: {source_path}")

    source_format = _detect_format(source_path)
    logger.info("Loading %s file: %s", source_format, source_path)
    try:
        if source_format == "csv":
            options = dict(read_options)
            if encoding is not None:
                options["encoding"] = encoding
            data = pl.read_csv(source_path, **options)
        elif source_format == "excel":
            if isinstance(sheet_name, str):
                data = pl.read_excel(source_path, sheet_name=sheet_name, **read_options)
            else:
                # Polars sheet_id is one-based; expose Python's familiar zero-based index.
                data = pl.read_excel(source_path, sheet_id=sheet_name + 1, **read_options)
        else:
            data = pl.read_parquet(source_path, **read_options)
    except Exception as exc:
        logger.exception("Failed to load file: %s", source_path)
        raise DataLoaderError(f"Failed to load '{source_path}': {exc}") from exc

    if not isinstance(data, pl.DataFrame):
        raise DataLoaderError(
            "Expected one table, but the reader returned multiple sheets. "
            "Pass a single Excel sheet name or index."
        )

    result = DataLoadResult(
        data=data,
        path=source_path,
        source_format=source_format,
        row_count=len(data),
        column_count=len(data.columns),
    )
    logger.info(
        "Loaded successfully: rows=%d, columns=%d, format=%s",
        result.row_count,
        result.column_count,
        result.source_format,
    )
    return result
