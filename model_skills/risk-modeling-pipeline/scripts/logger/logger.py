"""Centralized logging configuration for the risk-modeling project."""

from __future__ import annotations
import logging
from pathlib import Path
import sys

PROJECT_LOGGER_NAME = "risk_modeling"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    *,
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure the project logger once and return it.

    Write logs to stdout by default. When ``log_file`` is supplied, also append
    UTF-8 logs to that file. Repeated calls update the level without duplicating
    handlers.
    """
    project_logger = logging.getLogger(PROJECT_LOGGER_NAME)
    project_logger.setLevel(level)
    project_logger.propagate = False

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    if not project_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        project_logger.addHandler(console_handler)

    if log_file is not None:
        target_path = Path(log_file).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == target_path
            for handler in project_logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(target_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            project_logger.addHandler(file_handler)

    return project_logger


def get_logger(module_name: str) -> logging.Logger:
    """Return a namespaced logger for one project module."""
    configure_logging()
    return logging.getLogger(f"{PROJECT_LOGGER_NAME}.{module_name}")
