"""Standalone sample diagnosis and treatment-policy confirmation node.

The node diagnoses the source sample and records the user's selected policy.
It intentionally does not modify or persist a treated dataset. Later feature
preprocessing/modeling nodes apply the confirmed policy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ..common import NodeContext, resolve_path, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the sample-diagnosis workflow node")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--engine-root", help="Accepted for Skill command compatibility")
    parser.add_argument("--data")
    parser.add_argument("--data-contract")
    parser.add_argument("--sample-config")
    parser.add_argument("--node-config-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--confirm-config",
        action="store_true",
        help="Record the sample treatment policy after the user confirms the node YAML",
    )
    return parser


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_sample_config(args: argparse.Namespace, context: NodeContext) -> Path:
    """Resolve the single node config, with compatibility for old workspaces."""
    if args.sample_config:
        return resolve_path(args.sample_config, context.project_root)
    if "diagnostics" in context.config.parameters and "treatment" in context.config.parameters:
        return context.config.path
    legacy_path = context.project_root / "configs/sample_config.yaml"
    if legacy_path.is_file():
        return legacy_path
    return context.config.path


def run(args: argparse.Namespace) -> dict:
    context = NodeContext.from_args(args, "sample-diagnosis")
    from data.contract import validate_contract
    from preprocessing.sample_config import load_sample_config
    from preprocessing.sample_diagnostics import build_sample_diagnostics

    data_read_approval = context.node_config_dir / "data-read.approval.json"
    if not data_read_approval.is_file():
        raise RuntimeError(
            "data-read.yaml has not been confirmed. Complete the data-read node before starting sample diagnosis."
        )
    approval = json.loads(data_read_approval.read_text(encoding="utf-8"))
    current_read_hash = _hash(context.data_read_config.path)
    if approval.get("config_sha256") != current_read_hash:
        raise RuntimeError(
            "data-read.yaml changed after confirmation. Re-run data-read with --confirm-config before starting sample diagnosis."
        )

    eda_approval_path = context.node_config_dir / "eda-analysis.approval.json"
    if not eda_approval_path.is_file():
        raise RuntimeError(
            "EDA has not been confirmed and completed. Run the eda-analysis node with --confirm-config before starting sample diagnosis."
        )
    eda_approval = json.loads(eda_approval_path.read_text(encoding="utf-8"))
    eda_config_path = context.node_config_dir / "eda-analysis.yaml"
    if not eda_config_path.is_file():
        raise RuntimeError("eda-analysis.yaml is missing. Re-run the eda-analysis node before starting sample diagnosis.")
    eda_config_hash = _hash(eda_config_path)
    if eda_approval.get("config_sha256") != eda_config_hash:
        raise RuntimeError(
            "eda-analysis.yaml changed after confirmation. Re-run eda-analysis with --confirm-config before starting sample diagnosis."
        )

    sample_config_path = _resolve_sample_config(args, context)
    sample_config = load_sample_config(sample_config_path)
    treatment_policy = asdict(sample_config.treatment)
    with context.progress.track(
        "loader",
        running_summary="正在按已确认的数据配置重新读取数据",
        success_summary="样本诊断数据读取完成",
    ):
        loaded = context.reload_data()
        validated = validate_contract(
            loaded.data,
            context.effective_contract,
            allow_id_duplicates=True,
            allow_target_issues=True,
        )

    diagnostics = build_sample_diagnostics(loaded.data, validated.contract, sample_config)
    diagnostics_path = write_json(
        context.output_dir / "sample_diagnostics.json",
        {"node_id": "sample-diagnosis", **diagnostics},
    )
    confirmation = write_json(
        context.output_dir / "sample_diagnosis_confirmation.json",
        {
            "node_id": "sample-diagnosis",
            "status": "confirmed" if args.confirm_config else "awaiting_user_confirmation",
            "message": (
                "请确认 sample-diagnosis.yaml 中的诊断阈值和每一种样本处理方法。"
                "本节点只记录策略，不修改或保存处理后的样本，也不会切分数据或训练模型。"
                "其中 class_weight 仅作为后续训练提示。确认后重新执行并追加 --confirm-config。"
            ),
            "node_config_path": str(context.config.path),
            "node_config_sha256": _hash(context.config.path),
            "policy_config_path": str(sample_config_path),
            "policy_config_sha256": _hash(sample_config_path),
            "sample_config_path": str(sample_config_path),
            "sample_config_sha256": _hash(sample_config_path),
            "data_read_config_sha256": current_read_hash,
            "eda_config_sha256": eda_config_hash,
            "treatment_policy": treatment_policy,
            "data_path": str(context.data_path),
            "diagnostics": diagnostics,
            "diagnostics_artifact": str(diagnostics_path),
        },
    )
    if not args.confirm_config:
        context.progress.emit(
            "sample-diagnosis",
            "waiting_confirmation",
            summary="等待用户确认样本诊断和处理策略",
            artifacts=[diagnostics_path, confirmation],
        )
        return {
            "status": "awaiting_user_confirmation",
            "node_id": "sample-diagnosis",
            "diagnostics": str(diagnostics_path),
            "confirmation": str(confirmation),
        }

    policy_path = write_json(
        context.output_dir / "sample_treatment_policy.json",
        {
            "node_id": "sample-diagnosis",
            "status": "confirmed",
            "message": "已确认样本处理策略；本节点未修改或保存处理后的数据。",
            "policy_source": str(sample_config_path),
            "policy_source_sha256": _hash(sample_config_path),
            "policy_config_path": str(sample_config_path),
            "treatment_policy": treatment_policy,
            "diagnostics": diagnostics,
            "diagnostics_artifact": str(diagnostics_path),
        },
    )
    manifest = write_json(
        context.output_dir / "sample_diagnosis_manifest.json",
        {
            "node_id": "sample-diagnosis",
            "status": "success",
            "data_path": str(context.data_path),
            "data_read_config": str(context.data_read_config.path),
            "data_read_config_sha256": current_read_hash,
            "eda_config": str(eda_config_path),
            "eda_config_sha256": eda_config_hash,
            "node_config": str(context.config.path),
            "node_config_sha256": _hash(context.config.path),
            "policy_config": str(sample_config_path),
            "policy_config_sha256": _hash(sample_config_path),
            "sample_config": str(sample_config_path),
            "sample_config_sha256": _hash(sample_config_path),
            "treatment_policy": treatment_policy,
            "diagnostics": str(diagnostics_path),
            "policy": str(policy_path),
        },
    )
    node_approval = write_json(
        context.node_config_dir / "sample-diagnosis.approval.json",
        {
            "node_id": "sample-diagnosis",
            "status": "approved",
            "node_config": str(context.config.path),
            "node_config_sha256": _hash(context.config.path),
            "policy_config": str(sample_config_path),
            "policy_config_sha256": _hash(sample_config_path),
            "sample_config": str(sample_config_path),
            "sample_config_sha256": _hash(sample_config_path),
            "treatment_policy": treatment_policy,
            "policy_artifact": str(policy_path),
        },
    )
    context.progress.emit(
        "sample-diagnosis",
        "success",
        summary="样本诊断完成，处理策略已确认（未修改样本数据）",
        artifacts=[diagnostics_path, policy_path, manifest, node_approval],
    )
    return {
        "status": "success",
        "node_id": "sample-diagnosis",
        "diagnostics": str(diagnostics_path),
        "sample_treatment_policy": str(policy_path),
        "manifest": str(manifest),
        "approval": str(node_approval),
    }


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
