"""Independent rule-backed model review packet for AI explanation.

The reviewer consumes deterministic artifacts produced by the modeling
pipeline. It never invents a metric; the conversational agent may explain
these findings, but the evidence and severity remain rule-based.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from logger import get_logger

from .schemas import Finding, finding_dicts, report_status


logger = get_logger(__name__)


# Conservative defaults. They are returned in the evidence packet so callers
# can audit the exact rule version used for a run.
_AUC_GAP_WARNING = 0.08
_AUC_GAP_BLOCKER = 0.15
_KS_GAP_WARNING = 0.08
_KS_GAP_BLOCKER = 0.15
_LOW_AUC_WARNING = 0.60
_LOW_AUC_BLOCKER = 0.55
_LOW_KS_WARNING = 0.15
_LOW_KS_BLOCKER = 0.10
_OOT_DROP_WARNING = 0.05
_OOT_DROP_BLOCKER = 0.10
_PSI_WARNING = 0.10
_PSI_BLOCKER = 0.25
_MIN_RELIABLE_BAD_COUNT = 10
_MIN_MONTH_SAMPLE = 30


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _row_value(row: dict[str, Any] | None, key: str) -> float | None:
    return _number(row.get(key)) if row else None


def _severity_by_threshold(value: float, *, warning: float, blocker: float) -> str | None:
    if value >= blocker:
        return "blocker"
    if value >= warning:
        return "warning"
    return None


def _recommendations(findings: list[Finding]) -> list[dict[str, Any]]:
    """Turn findings into deduplicated, actionable remediation items."""
    actions: dict[str, dict[str, Any]] = {}
    mapping = {
        "overfit": (
            "优先",
            "收紧模型复杂度（num_leaves/max_depth/min_data_in_leaf）、减少不稳定特征，并使用 Train-only CV 重新调参；同时复核泄漏字段。",
        ),
        "underfit": (
            "优先",
            "检查过严的缺失率/IV/相关性筛选和字符变量策略，补充观察期内有效特征，再在参数边界内逐步增加模型表达能力。",
        ),
        "oot": (
            "优先",
            "复核观察期和 OOT 时间窗口，扩大 OOT 坏样本量或补充近期样本；不得用 OOT 反复调参。",
        ),
        "drift": (
            "优先",
            "定位发生漂移的分数/特征及业务口径变化，检查数据源和时间切分；必要时使用近期稳定样本重训并重新确认窗口。",
        ),
        "lift": (
            "建议",
            "检查标签定义、排序方向、样本量和分箱方式；若排序性持续不佳，重新评估特征质量、模型目标和阈值策略。",
        ),
        "importance": (
            "建议",
            "对高贡献特征做单变量稳定性、时间穿越和业务可解释性复核，必要时剔除或设置单特征贡献约束。",
        ),
        "monthly": (
            "建议",
            "拆分波动月份检查样本量、标签口径和业务策略变化；稳定性不足时不要直接宣称模型可上线。",
        ),
    }
    for finding in findings:
        code = finding.code.lower()
        for category, (priority, action) in mapping.items():
            message = finding.message
            matched = category in code or category in message
            matched = matched or (
                category == "overfit" and ("过拟合" in message or "泛化能力" in message)
            )
            matched = matched or (category == "underfit" and "欠拟合" in message)
            matched = matched or (category == "drift" and "漂移" in message)
            if not matched:
                continue
            item = actions.setdefault(
                category,
                {
                    "priority": priority,
                    "category": category,
                    "source_codes": [],
                    "action": action,
                },
            )
            item["source_codes"].append(finding.code)
            break
    return list(actions.values())


def _lift_findings(lift_detail: pl.DataFrame | None) -> list[Finding]:
    findings: list[Finding] = []
    if lift_detail is None or lift_detail.is_empty():
        return findings
    if not {"dataset", "bucket", "bad_rate"}.issubset(lift_detail.columns):
        return findings
    for dataset, frame in lift_detail.partition_by("dataset", as_dict=True).items():
        dataset = dataset[0] if isinstance(dataset, tuple) else dataset
        ordered = frame.sort("bucket")
        rates = [_number(value) for value in ordered.get_column("bad_rate").to_list()]
        rates = [value for value in rates if value is not None]
        if len(rates) < 5:
            continue
        violations = sum(left < right for left, right in zip(rates, rates[1:]))
        allowed = max(1, int((len(rates) - 1) * 0.30))
        top_lift = _number(ordered.get_column("lift")[0]) if "lift" in ordered.columns else None
        if violations > allowed or (top_lift is not None and top_lift < 1.0):
            findings.append(
                Finding(
                    "LIFT_ORDERING_WEAK",
                    "warning",
                    f"{dataset} 风险分箱坏样本率排序性不够稳定，存在 {violations} 个逆序相邻分箱。",
                    {
                        "dataset": dataset,
                        "bucket_count": len(rates),
                        "monotonicity_violations": violations,
                        "allowed_violations": allowed,
                        "top_lift": top_lift,
                    },
                    "检查标签方向、样本量、分箱方式和特征质量；必要时重新训练或调整评分分层策略。",
                )
            )
    return findings


def _monthly_findings(monthly_performance: pl.DataFrame | None) -> list[Finding]:
    findings: list[Finding] = []
    required = {"dataset", "sample_count", "bad_count"}
    if monthly_performance is None or monthly_performance.is_empty():
        return findings
    if not required.issubset(monthly_performance.columns):
        return findings
    for dataset, frame in monthly_performance.partition_by("dataset", as_dict=True).items():
        dataset = dataset[0] if isinstance(dataset, tuple) else dataset
        reliable = frame.filter(
            (pl.col("sample_count") >= _MIN_MONTH_SAMPLE)
            & (pl.col("bad_count") >= _MIN_RELIABLE_BAD_COUNT)
        )
        if reliable.height < 3:
            continue
        for metric in ("auc", "ks", "bad_rate"):
            if metric not in reliable.columns:
                continue
            values = [_number(value) for value in reliable.get_column(metric).to_list()]
            values = [value for value in values if value is not None]
            if len(values) < 3:
                continue
            spread = max(values) - min(values)
            threshold = 0.10 if metric in {"auc", "ks"} else 0.05
            if spread >= threshold:
                findings.append(
                    Finding(
                        "MONTHLY_PERFORMANCE_UNSTABLE",
                        "warning",
                        f"{dataset} 月度 {metric.upper()} 波动较大，最大差值为 {spread:.4f}。",
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "month_count": len(values),
                            "min": min(values),
                            "max": max(values),
                            "spread": spread,
                        },
                        "拆分波动月份核对样本量、标签口径和业务策略变化，确认时间切分和近期样本代表性。",
                    )
                )
                break
    return findings


def build_model_review(
    metrics: pl.DataFrame,
    *,
    best_iteration: int,
    feature_importance: pl.DataFrame,
    training_history: pl.DataFrame | None = None,
    lift_detail: pl.DataFrame | None = None,
    monthly_performance: pl.DataFrame | None = None,
    split_summary: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Identify model risks and produce evidence-backed improvements."""
    by_split = {str(row["dataset"]): row for row in metrics.to_dicts()}
    findings: list[Finding] = []
    train, test, oot = by_split.get("train"), by_split.get("test"), by_split.get("oot")
    train_auc, test_auc = _row_value(train, "auc"), _row_value(test, "auc")
    train_ks, test_ks = _row_value(train, "ks"), _row_value(test, "ks")

    if train_auc is not None and test_auc is not None:
        auc_gap = train_auc - test_auc
        severity = _severity_by_threshold(auc_gap, warning=_AUC_GAP_WARNING, blocker=_AUC_GAP_BLOCKER)
        if severity:
            findings.append(
                Finding(
                    "TRAIN_TEST_AUC_GAP_HIGH" if severity == "warning" else "TRAIN_TEST_AUC_GAP_CRITICAL",
                    severity,
                    f"Train/Test AUC 差距为 {auc_gap:.4f}，存在明显过拟合风险。",
                    {"train_auc": train_auc, "test_auc": test_auc, "auc_gap": auc_gap},
                    "收紧模型复杂度、检查泄漏并重新调参。",
                )
            )
    if train_ks is not None and test_ks is not None:
        ks_gap = train_ks - test_ks
        severity = _severity_by_threshold(ks_gap, warning=_KS_GAP_WARNING, blocker=_KS_GAP_BLOCKER)
        if severity:
            findings.append(
                Finding(
                    "TRAIN_TEST_KS_GAP_HIGH" if severity == "warning" else "TRAIN_TEST_KS_GAP_CRITICAL",
                    severity,
                    f"Train/Test KS 差距为 {ks_gap:.4f}，泛化能力存在风险。",
                    {"train_ks": train_ks, "test_ks": test_ks, "ks_gap": ks_gap},
                    "检查模型复杂度、特征泄漏和 Test 样本代表性，并使用 Train-only CV 重新评估。",
                )
            )

    if train_auc is not None and test_auc is not None and train_ks is not None and test_ks is not None:
        if train_auc < _LOW_AUC_BLOCKER and test_auc < _LOW_AUC_BLOCKER:
            findings.append(
                Finding(
                    "MODEL_UNDERFIT_CRITICAL",
                    "blocker",
                    "Train 和 Test 的 AUC 均偏低，模型区分能力不足，疑似欠拟合或特征信息不足。",
                    {"train_auc": train_auc, "test_auc": test_auc, "train_ks": train_ks, "test_ks": test_ks},
                    "检查目标标签、特征筛选和字符变量处理；补充有效特征后再调整模型表达能力。",
                )
            )
        elif (
            train_auc < _LOW_AUC_WARNING and test_auc < _LOW_AUC_WARNING
        ) or (train_ks < _LOW_KS_WARNING and test_ks < _LOW_KS_WARNING):
            findings.append(
                Finding(
                    "MODEL_UNDERFIT_WARNING",
                    "warning",
                    "Train 和 Test 的区分指标整体偏低，模型可能欠拟合或输入特征信息不足。",
                    {"train_auc": train_auc, "test_auc": test_auc, "train_ks": train_ks, "test_ks": test_ks},
                    "复核特征质量和筛选阈值，确认标签定义与观察窗口，再在边界内调整模型复杂度。",
                )
            )

    if test and oot:
        for metric in ("auc", "ks"):
            test_value, oot_value = _row_value(test, metric), _row_value(oot, metric)
            if test_value is None or oot_value is None:
                continue
            drop = test_value - oot_value
            severity = _severity_by_threshold(drop, warning=_OOT_DROP_WARNING, blocker=_OOT_DROP_BLOCKER)
            if severity:
                findings.append(
                    Finding(
                        f"OOT_{metric.upper()}_DROP",
                        severity,
                        f"OOT {metric.upper()} 较 Test 下降 {drop:.4f}，时间外泛化能力不足。",
                        {"test": test_value, "oot": oot_value, "drop": drop},
                        "复核时间窗口、近期样本代表性和数据口径；不要使用 OOT 反复调参。",
                    )
                )

    if oot:
        oot_bad_count = int(oot.get("bad_count") or 0)
        if oot_bad_count < _MIN_RELIABLE_BAD_COUNT:
            findings.append(
                Finding(
                    "OOT_EVALUATION_UNSUPPORTED",
                    "blocker",
                    f"OOT 坏样本仅 {oot_bad_count} 个，不支持稳定的生产准入结论。",
                    {"oot_bad_count": oot_bad_count, "minimum_reliable_bad_count": _MIN_RELIABLE_BAD_COUNT},
                    "扩大 OOT 时间窗口或补充近期样本，并重新确认时间切分方案。",
                )
            )

    for dataset in ("test", "oot"):
        row = by_split.get(dataset)
        psi = _row_value(row, "score_psi_vs_train")
        if psi is None:
            continue
        severity = _severity_by_threshold(psi, warning=_PSI_WARNING, blocker=_PSI_BLOCKER)
        if severity:
            findings.append(
                Finding(
                    f"{dataset.upper()}_SCORE_PSI_HIGH",
                    severity,
                    f"{dataset.upper()} 相对 Train 的分数 PSI 为 {psi:.4f}，存在样本/分数分布漂移。",
                    {"dataset": dataset, "score_psi_vs_train": psi, "warning": _PSI_WARNING, "blocker": _PSI_BLOCKER},
                    "定位漂移特征和数据源口径，复核近期样本并评估是否需要窗口更新或重训。",
                )
            )

    if best_iteration <= 10:
        findings.append(
            Finding(
                "EARLY_STOP_TOO_EARLY",
                "warning",
                "最佳迭代轮数不超过 10，需检查过拟合、学习率和验证样本波动。",
                {"best_iteration": best_iteration},
                "检查学习率、验证集规模和模型复杂度，结合训练曲线判断是否需要重新切分。",
            )
        )

    if training_history is not None and not training_history.is_empty():
        if {"train_auc", "test_auc"}.issubset(training_history.columns):
            gaps = [
                _number(row["train_auc"]) - _number(row["test_auc"])
                for row in training_history.to_dicts()
                if _number(row.get("train_auc")) is not None and _number(row.get("test_auc")) is not None
            ]
            if gaps and max(gaps) - min(gaps) >= 0.10:
                findings.append(
                    Finding(
                        "TRAINING_GAP_WIDENS",
                        "warning",
                        "训练过程中 Train/Test AUC gap 明显扩大，模型复杂度或验证样本波动可能在加剧。",
                        {"min_auc_gap": min(gaps), "max_auc_gap": max(gaps), "gap_range": max(gaps) - min(gaps)},
                        "结合最佳迭代轮数收紧早停和模型复杂度，避免只看最终单点指标。",
                    )
                )

    if feature_importance.height and "gain_importance" in feature_importance.columns:
        total_gain = float(feature_importance.get_column("gain_importance").sum() or 0.0)
        top = feature_importance.row(0, named=True)
        top_share = float(top["gain_importance"]) / total_gain if total_gain else 0.0
        if top_share > 0.5:
            findings.append(
                Finding(
                    "FEATURE_IMPORTANCE_CONCENTRATED",
                    "warning",
                    "单一特征贡献超过总增益的 50%，需检查泄漏和稳定性。",
                    {"feature": top["feature"], "gain_share": top_share},
                    "对高贡献特征做时间穿越、单变量稳定性和业务可解释性复核。",
                )
            )

    findings.extend(_lift_findings(lift_detail))
    findings.extend(_monthly_findings(monthly_performance))
    recommendations = _recommendations(findings)
    status = report_status(findings)
    logger.info(
        "Model review completed: status=%s, findings=%d, recommendations=%d",
        status,
        len(findings),
        len(recommendations),
    )
    evidence = {
        "split_metrics": metrics.to_dicts(),
        "thresholds": {
            "auc_gap_warning": _AUC_GAP_WARNING,
            "auc_gap_blocker": _AUC_GAP_BLOCKER,
            "ks_gap_warning": _KS_GAP_WARNING,
            "ks_gap_blocker": _KS_GAP_BLOCKER,
            "oot_drop_warning": _OOT_DROP_WARNING,
            "oot_drop_blocker": _OOT_DROP_BLOCKER,
            "score_psi_warning": _PSI_WARNING,
            "score_psi_blocker": _PSI_BLOCKER,
            "minimum_oot_bad_count": _MIN_RELIABLE_BAD_COUNT,
        },
        "has_training_history": training_history is not None and not training_history.is_empty(),
        "has_lift_detail": lift_detail is not None and not lift_detail.is_empty(),
        "has_monthly_performance": monthly_performance is not None and not monthly_performance.is_empty(),
        "has_split_summary": split_summary is not None and not split_summary.is_empty(),
    }
    return {
        "status": status,
        "findings": finding_dicts(findings),
        "diagnosis": finding_dicts(findings),
        "recommendations": recommendations,
        "evidence": evidence,
        "production_readiness": "not_approved" if findings else "requires_human_approval",
        "requires_user_confirmation": True,
    }
