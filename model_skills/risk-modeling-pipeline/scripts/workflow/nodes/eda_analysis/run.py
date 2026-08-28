"""Run the confirmed EDA analysis node.

The node deliberately reloads the source data in a fresh process.  It reads
the approved data-read configuration for roles/loader options and its own EDA
configuration for binning, stability and report settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..common import NodeContext, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the EDA analysis workflow node")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--engine-root", help="Accepted for Skill command compatibility")
    parser.add_argument("--data")
    parser.add_argument("--data-contract")
    parser.add_argument("--node-config-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--confirm-config",
        action="store_true",
        help="Execute after the user has reviewed and confirmed data-read.yaml and eda-analysis.yaml",
    )
    return parser


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    context = NodeContext.from_args(args, "eda-analysis")
    from data.contract import validate_contract

    data_read_approval = context.node_config_dir / "data-read.approval.json"
    if not data_read_approval.is_file():
        raise RuntimeError(
            "data-read.yaml has not been explicitly confirmed. Run the data-read "
            "node with --confirm-config before starting EDA."
        )
    approval = json.loads(data_read_approval.read_text(encoding="utf-8"))
    current_data_read_hash = _hash(context.data_read_config.path)
    if approval.get("config_sha256") != current_data_read_hash:
        raise RuntimeError(
            "data-read.yaml changed after confirmation. Re-run data-read with "
            "--confirm-config before starting EDA."
        )

    # This is intentionally a fresh read.  No dataframe is inherited from the
    # data-read node, so the EDA evidence always reflects the current YAML.
    with context.progress.track(
        "loader",
        running_summary="正在按已确认的数据配置重新读取数据",
        success_summary="数据重新读取完成，等待 EDA 配置确认",
    ):
        loaded = context.reload_data()
        validated = validate_contract(
            loaded.data,
            context.effective_contract,
            allow_id_duplicates=True,
            allow_target_issues=True,
        )

    eda_config_path = context.config.path
    data_read_config_path = context.data_read_config.path
    confirmation = write_json(
        context.output_dir / "eda_confirmation.json",
        {
            "node_id": "eda-analysis",
            "status": "awaiting_user_confirmation",
            "message": "数据读取配置已确认。请查看 configs/node_configs/eda-analysis.yaml 并确认 EDA 参数（分箱、PSI 基准月、质量阈值和报告选项）；当前节点不执行 Train/Test/OOT 切分、class_weight 或模型训练。确认后重新执行并追加 --confirm-config。",
            "project_root": str(context.project_root),
            "data_path": str(context.data_path),
            "data_read_config_path": str(data_read_config_path),
            "data_read_config_sha256": _hash(data_read_config_path),
            "eda_config_path": str(eda_config_path),
            "eda_config_sha256": _hash(eda_config_path),
            "row_count": loaded.row_count,
            "column_count": loaded.column_count,
            "feature_count": len(validated.feature_cols),
            "parameters": context.config.parameters,
            "display_files": [str(eda_config_path)],
        },
    )
    if not args.confirm_config:
        context.progress.emit(
            "eda-analysis",
            "waiting_confirmation",
            summary="等待用户确认 EDA 分析配置",
            artifacts=[confirmation],
        )
        return {
            "status": "awaiting_user_confirmation",
            "node_id": "eda-analysis",
            "confirmation": str(confirmation),
            "config": str(eda_config_path),
            "display_files": [str(eda_config_path)],
        }

    from eda.pipeline import run_eda

    parameters = context.config.parameters
    with context.progress.track(
        "eda",
        running_summary="正在计算字段画像、分箱、IV/KS、PSI、相关性和月度分析",
        success_summary="EDA 分析计算完成",
        artifacts=[context.output_dir / "eda_summary.json"],
    ):
        result = run_eda(
            loaded.data,
            validated,
            context.output_dir,
            progress=context.progress,
            report_dir=context.output_dir / "reports",
            generate_report=bool(parameters.get("generate_report", True)),
            analysis_config=parameters,
        )

    manifest = write_json(
        context.output_dir / "eda_manifest.json",
        {
            "node_id": "eda-analysis",
            "status": "success",
            "project_root": str(context.project_root),
            "data_path": str(context.data_path),
            "data_read_config": str(data_read_config_path),
            "data_read_config_sha256": _hash(data_read_config_path),
            "eda_config": str(eda_config_path),
            "eda_config_sha256": _hash(eda_config_path),
            "report_path": str(result.report_path),
            "cleaned_data_path": str(result.cleaned_data_path),
            "output_dir": str(context.output_dir),
        },
    )
    # Persist a hash-bound gate for downstream standalone nodes. This keeps
    # sample diagnosis from being started before the confirmed EDA run, while
    # allowing the two nodes to use different output directories/processes.
    approval = write_json(
        context.node_config_dir / "eda-analysis.approval.json",
        {
            "node_id": "eda-analysis",
            "status": "approved",
            "config_path": str(eda_config_path),
            "config_sha256": _hash(eda_config_path),
            "data_read_config_path": str(data_read_config_path),
            "data_read_config_sha256": _hash(data_read_config_path),
            "manifest_path": str(manifest),
        },
    )
    context.progress.emit(
        "eda-analysis",
        "success",
        summary="EDA 分析完成，配置已锁定供后续节点使用",
        artifacts=[manifest, approval],
    )
    return {
        "status": "success",
        "node_id": "eda-analysis",
        "report": str(result.report_path),
        "manifest": str(manifest),
        "approval": str(approval),
    }


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") in {"success", "awaiting_user_confirmation"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
