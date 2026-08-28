"""指标计算安全封装。"""

from __future__ import annotations

import logging

import pandas as pd
import toad
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def safe_ks(series: pd.Series, target: pd.Series, default: float = 0.0) -> float:
    clean = pd.DataFrame({"x": series, "y": target}).dropna()
    if len(clean) == 0:
        return default
    try:
        return float(toad.metrics.KS(clean["x"], clean["y"]))
    except Exception:
        return default


def safe_psi(expected: pd.Series, actual: pd.Series) -> float:
    exp = expected.dropna()
    act = actual.dropna()
    if len(exp) < 2 or len(act) < 2:
        return float("nan")
    try:
        return float(toad.metrics.PSI(exp, act))
    except Exception as exc:
        logger.warning("PSI 计算失败: %s", exc)
        return float("nan")


def safe_auc(target: pd.Series, score: pd.Series) -> float:
    clean = pd.DataFrame({"x": score, "y": target}).dropna()
    if len(clean) < 2 or clean["y"].nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(clean["y"], clean["x"]))
    except Exception:
        return float("nan")
