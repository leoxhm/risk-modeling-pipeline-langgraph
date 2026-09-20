"""轻量的二分类评估指标，避免依赖旧版建模引擎。"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import toad


def model_quality_components(metrics: dict[str, Any]) -> dict[str, float | None]:
    """计算用于最终模型选择的综合质量分及其组成项。

    风控模型不能只看 Validate KS：Train 过高而 Validate/OOT 明显下降通常
    是过拟合，OOT 较低则说明时间外推能力不足。因此这里对 OOT 赋更高权重，
    同时惩罚 Train-Validate 和 Validate-OOT 的差距；若日志中有 OOT Score PSI，
    还会对明显漂移做轻度扣分。最终分数被限制在 0~1，便于面板和报告解释。
    """
    values: dict[str, float | None] = {}
    for dataset in ("train", "validate", "oot"):
        raw = (metrics.get(dataset) or {}).get("ks") if isinstance(metrics, dict) else None
        try:
            values[dataset] = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            values[dataset] = None
    if any(values.get(dataset) is None for dataset in ("train", "validate", "oot")):
        return {
            "quality_score": None,
            "weighted_ks": None,
            "train_validate_gap": None,
            "validate_oot_gap": None,
            "oot_psi_penalty": None,
        }
    train_ks = float(values["train"])
    validate_ks = float(values["validate"])
    oot_ks = float(values["oot"])
    # OOT 是最接近未来生产样本的验证集，权重最高。
    weighted_ks = 0.25 * train_ks + 0.30 * validate_ks + 0.45 * oot_ks
    train_validate_gap = abs(train_ks - validate_ks)
    validate_oot_gap = abs(validate_ks - oot_ks)
    # gap 惩罚比单纯取平均更能排除“Train 很高、OOT 很差”的模型。
    gap_penalty = 0.25 * train_validate_gap + 0.35 * validate_oot_gap
    oot_floor_penalty = 0.20 * max(0.0, 0.20 - oot_ks)
    raw_psi = (metrics.get("oot") or {}).get("score_psi_vs_train") if isinstance(metrics, dict) else None
    try:
        oot_psi = float(raw_psi) if raw_psi is not None else None
    except (TypeError, ValueError):
        oot_psi = None
    psi_penalty = 0.05 * max(0.0, oot_psi - 0.25) if oot_psi is not None else 0.0
    score = max(0.0, min(1.0, weighted_ks - gap_penalty - oot_floor_penalty - psi_penalty))
    return {
        "quality_score": score,
        "weighted_ks": weighted_ks,
        "train_validate_gap": train_validate_gap,
        "validate_oot_gap": validate_oot_gap,
        "oot_psi_penalty": psi_penalty,
    }


def model_quality_score(metrics: dict[str, Any]) -> float | None:
    """返回综合模型质量分；KS 缺失时返回 None。"""
    return model_quality_components(metrics).get("quality_score")


def auc_ks(labels: Iterable[int], scores: Iterable[float]) -> tuple[float | None, float | None]:
    """计算方向无关 AUC 和 KS；样本只有单一标签时返回空值。"""
    pairs = sorted(((float(score), int(label)) for score, label in zip(scores, labels)), key=lambda item: item[0])
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None, None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    ordered = sorted(pairs, key=lambda item: item[0], reverse=True)
    bad_seen = good_seen = 0
    ks = 0.0
    # KS 只能在“分数阈值”变化的位置计算。若所有样本分数相同，
    # 不能按样本逐个累计，否则会把稳定的 0.5 AUC 误判成 KS=1。
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        bad_seen += sum(label for _, label in ordered[index:end])
        good_seen += (end - index) - sum(label for _, label in ordered[index:end])
        ks = max(ks, abs(bad_seen / positives - good_seen / negatives))
        index = end
    return max(auc, 1.0 - auc), ks


def pr_auc(labels: Iterable[int], scores: Iterable[float]) -> float | None:
    """计算二分类 PR-AUC（Average Precision），适用于坏样本比例较低的场景。"""
    pairs = sorted(((float(score), int(label)) for score, label in zip(scores, labels)), key=lambda item: item[0], reverse=True)
    positives = sum(label for _, label in pairs)
    if not pairs or not positives:
        return None
    true_positive = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(pairs, start=1):
        true_positive += label
        if label:
            precision_sum += true_positive / index
    return precision_sum / positives


def score_psi(reference_scores: Iterable[float], current_scores: Iterable[float], bins: int = 10) -> float | None:
    """使用 Train 分数分箱并调用 toad 计算当前分区的 Score PSI。"""
    reference = np.asarray([float(value) for value in reference_scores if value is not None and math.isfinite(float(value))], dtype=float)
    current = np.asarray([float(value) for value in current_scores if value is not None and math.isfinite(float(value))], dtype=float)
    if not len(reference) or not len(current):
        return None
    if np.all(reference == reference[0]) and np.all(current == current[0]) and reference[0] == current[0]:
        return 0.0
    cuts = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    reference_bins = np.digitize(reference, cuts, right=True)
    current_bins = np.digitize(current, cuts, right=True)
    try:
        return float(toad.metrics.PSI(current_bins, reference_bins))
    except (TypeError, ValueError, IndexError):
        # 极端常量分布下 toad 可能无法建桶，使用同样分桶的平滑 PSI 兜底。
        categories = max(len(cuts) + 1, 1)
        reference_count = np.bincount(reference_bins, minlength=categories).astype(float)
        current_count = np.bincount(current_bins, minlength=categories).astype(float)
        reference_rate = (reference_count + 1e-6) / (reference_count.sum() + 1e-6 * categories)
        current_rate = (current_count + 1e-6) / (current_count.sum() + 1e-6 * categories)
        return float(np.sum((current_rate - reference_rate) * np.log(current_rate / reference_rate)))


def _sample_points(points: list[dict[str, float]], limit: int = 41) -> list[dict[str, float]]:
    """压缩曲线点数，保证 JSONL 可持续追加且报告仍保持曲线形状。"""
    if len(points) <= limit:
        return points
    indexes = np.linspace(0, len(points) - 1, limit, dtype=int)
    return [points[int(index)] for index in indexes]


def classification_curves(labels: Iterable[int], scores: Iterable[float]) -> dict[str, list[dict[str, float]]]:
    """生成 ROC、PR 曲线点，供报告绘图，不依赖 sklearn。"""
    pairs = sorted(((float(score), int(label)) for score, label in zip(scores, labels)), key=lambda item: item[0], reverse=True)
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not pairs or not positives or not negatives:
        return {"roc_curve": [], "pr_curve": []}
    tp = fp = 0
    roc = [{"fpr": 0.0, "tpr": 0.0}]
    pr: list[dict[str, float]] = [{"recall": 0.0, "precision": 1.0}]
    index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == threshold:
            tp += pairs[index][1]
            fp += 1 - pairs[index][1]
            index += 1
        recall = tp / positives
        precision = tp / (tp + fp) if tp + fp else 1.0
        roc.append({"fpr": fp / negatives, "tpr": recall})
        pr.append({"recall": recall, "precision": precision})
    return {"roc_curve": _sample_points(roc), "pr_curve": _sample_points(pr)}


def decile_analysis(labels: Iterable[int], scores: Iterable[float], groups: int = 10) -> list[dict[str, float | int]]:
    """按预测分数降序生成十分位坏账率、Lift 和累计 Gain。"""
    pairs = sorted(((float(score), int(label)) for score, label in zip(scores, labels)), key=lambda item: item[0], reverse=True)
    if not pairs:
        return []
    total = len(pairs)
    total_bad = sum(label for _, label in pairs)
    overall_bad_rate = total_bad / total if total else 0.0
    result: list[dict[str, float | int | str]] = []
    cumulative_bad = 0
    for index in range(groups):
        start = index * total // groups
        end = (index + 1) * total // groups if index < groups - 1 else total
        chunk = pairs[start:end]
        if not chunk:
            continue
        bad_count = sum(label for _, label in chunk)
        cumulative_bad += bad_count
        bad_rate = bad_count / len(chunk)
        result.append(
            {
                "decile": index + 1,
                "sample_count": len(chunk),
                "bad_count": bad_count,
                "bad_rate": bad_rate,
                "lift": bad_rate / overall_bad_rate if overall_bad_rate else 0.0,
                "cumulative_gain": cumulative_bad / total_bad if total_bad else 0.0,
                "sample_rate": len(chunk) / total,
            }
        )
    return result


def score_distribution(labels: Iterable[int], scores: Iterable[float], bins: int = 10) -> list[dict[str, float | int | str]]:
    """按预测概率从低到高切成等量分位，统计好坏样本数量。

    固定使用 0.1 概率区间会让低坏账率数据全部挤在 0.0-0.1，
    图形无法比较。因此这里改成分数十分位，并同时记录每个分位的分数范围。
    """
    result: list[dict[str, float | int]] = []
    pairs = sorted(((float(score), int(label)) for score, label in zip(scores, labels)), key=lambda item: item[0])
    if not pairs:
        return result
    total = len(pairs)
    for index in range(min(bins, total)):
        start = index * total // bins
        end = (index + 1) * total // bins if index < bins - 1 else total
        selected = pairs[start:end]
        if not selected:
            continue
        lower, upper = selected[0][0], selected[-1][0]
        result.append(
            {
                "bin": f"P{index + 1:02d} ({lower:.4f}–{upper:.4f})",
                "good_count": sum(1 for label, _ in selected if not label),
                "bad_count": sum(1 for label, _ in selected if label),
                "sample_count": len(selected),
                "score_min": lower,
                "score_max": upper,
            }
        )
    return result
