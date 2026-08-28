"""Configuration-driven LightGBM modeling workflow.

The public configuration objects are imported eagerly, while the training
pipeline is loaded lazily.  This keeps data/feature-preprocessing checks
usable in environments where optional model-runtime dependencies (LightGBM
and Optuna) have not been installed yet.
"""

from .config import ModelConfig, load_model_config

__all__ = [
    "DataSplitResult",
    "ModelConfig",
    "ModelRunResult",
    "TuningResult",
    "load_model_config",
    "run_model_pipeline",
    "split_dataset",
    "tune_lightgbm",
]


def __getattr__(name: str):
    if name in {"ModelRunResult", "run_model_pipeline"}:
        from .pipeline import ModelRunResult, run_model_pipeline

        return {"ModelRunResult": ModelRunResult, "run_model_pipeline": run_model_pipeline}[name]
    if name in {"DataSplitResult", "split_dataset"}:
        from .split import DataSplitResult, split_dataset

        return {"DataSplitResult": DataSplitResult, "split_dataset": split_dataset}[name]
    if name in {"TuningResult", "tune_lightgbm"}:
        from .tuning import TuningResult, tune_lightgbm

        return {"TuningResult": TuningResult, "tune_lightgbm": tune_lightgbm}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
