"""Model artifact exporters.

The Python pickle is intended for trusted Python runtimes. PMML export is an
optional integration because the JPMML-LightGBM converter is a Java artifact
and is not part of the core Python environment.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
from typing import Any

import lightgbm as lgb

from logger import get_logger


logger = get_logger(__name__)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _write_status(path: Path, status: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def export_model_artifacts(
    booster: lgb.Booster,
    run_dir: str | Path,
    *,
    best_iteration: int,
    feature_cols: tuple[str, ...],
    preprocessing_plan: Any | None = None,
    model_config: Any | None = None,
    contract: Any | None = None,
    pmml_converter: str | Path | None = None,
) -> dict[str, str | None]:
    """Save native, pickle and optionally PMML model artifacts."""
    output = Path(run_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    native_path = output / "lightgbm_model.txt"
    pickle_path = output / "lightgbm_model.pkl"
    bundle_path = output / "model_bundle.pkl"
    booster.save_model(str(native_path), num_iteration=best_iteration)
    # Pickle the same best-iteration model that was written to the native
    # artifact, so a consumer calling ``predict`` without an explicit
    # ``num_iteration`` does not accidentally use early-stopped trees.
    pickle_booster = lgb.Booster(
        model_str=booster.model_to_string(num_iteration=best_iteration)
    )
    with pickle_path.open("wb") as handle:
        pickle.dump(pickle_booster, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bundle = {
        "booster": pickle_booster,
        "best_iteration": best_iteration,
        "feature_cols": feature_cols,
        "preprocessing_plan": preprocessing_plan,
        "model_config": model_config,
        "contract": contract,
    }
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    converter = pmml_converter or os.environ.get("JPMML_LIGHTGBM_JAR")
    pmml_path = output / "lightgbm_model.pmml"
    status_path = output / "pmml_export_status.json"
    if not converter:
        _write_status(
            status_path,
            {
                "status": "not_configured",
                "message": "未配置 JPMML-LightGBM 转换器；已保留 lightgbm_model.txt 供后续转换。",
                "native_model": str(native_path),
            },
        )
        logger.info("PMML converter not configured; native model retained: %s", native_path)
        return {
            "native": str(native_path),
            "pickle": str(pickle_path),
            "bundle": str(bundle_path),
            "pmml": None,
            "pmml_status": str(status_path),
        }

    converter_path = Path(converter).expanduser().resolve()
    java = shutil.which("java")
    if java is None or not converter_path.is_file():
        message = "找不到 Java 运行时或 JPMML-LightGBM JAR，跳过 PMML 生成。"
        _write_status(
            status_path,
            {
                "status": "unavailable",
                "message": message,
                "java": java,
                "converter": str(converter_path),
                "native_model": str(native_path),
            },
        )
        logger.warning("%s converter=%s", message, converter_path)
        return {
            "native": str(native_path),
            "pickle": str(pickle_path),
            "bundle": str(bundle_path),
            "pmml": None,
            "pmml_status": str(status_path),
        }

    command = [
        java,
        "-jar",
        str(converter_path),
        "--lgbm-input",
        str(native_path),
        "--pmml-output",
        str(pmml_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if not pmml_path.is_file():
            raise RuntimeError("JPMML converter exited successfully but did not create the PMML file")
        _write_status(
            status_path,
            {
                "status": "success",
                "converter": str(converter_path),
                "command": command,
                "stdout": completed.stdout[-2000:],
                "pmml_model": str(pmml_path),
            },
        )
        logger.info("PMML model created: %s", pmml_path)
        return {
            "native": str(native_path),
            "pickle": str(pickle_path),
            "bundle": str(bundle_path),
            "pmml": str(pmml_path),
            "pmml_status": str(status_path),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        if pmml_path.exists():
            pmml_path.unlink()
        _write_status(
            status_path,
            {
                "status": "failed",
                "message": str(exc),
                "converter": str(converter_path),
                "command": command,
                "native_model": str(native_path),
            },
        )
        logger.warning("PMML export failed; native and pickle models retained: %s", exc)
        return {
            "native": str(native_path),
            "pickle": str(pickle_path),
            "bundle": str(bundle_path),
            "pmml": None,
            "pmml_status": str(status_path),
        }
