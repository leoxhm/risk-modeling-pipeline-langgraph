"""Standalone data-read workflow node."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib

from ..common import NodeContext, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the data-read workflow node")
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
        help="Execute after the user has reviewed and confirmed data-read.yaml",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    context = NodeContext.from_args(args, "data-read")
    with context.progress.track(
        "loader",
        running_summary="正在读取数据与校验数据契约",
        success_summary="数据读取完成，等待配置确认",
        artifacts=[context.output_dir / "data_read_summary.json"],
    ):
        loaded = context.reload_data()
        from data.contract import apply_role_overrides, validate_contract
        from workflow.node_config import ensure_node_configs, populate_data_read_roles

        # Bootstrap only missing roles. This keeps the node self-contained:
        # a direct node invocation can read the source, write suggestions to
        # data-read.yaml, and validate the effective contract in one run.
        if populate_data_read_roles(
            context.data_read_config, loaded.data, context.contract
        ):
            refreshed = ensure_node_configs(context.node_config_dir, ["data-read"])[
                "data-read"
            ]
            context = replace(
                context,
                config=refreshed,
                data_read_config=refreshed,
                effective_contract=apply_role_overrides(
                    context.contract, refreshed.parameters
                ),
            )

        config_hash = hashlib.sha256(
            context.config.path.read_bytes()
        ).hexdigest()

        result = validate_contract(
            loaded.data,
            context.effective_contract,
            allow_id_duplicates=bool(
                context.config.parameters.get("allow_id_duplicates", False)
            ),
            allow_target_issues=bool(
                context.config.parameters.get("allow_target_issues", False)
            ),
        )
        artifact = write_json(
            context.output_dir / "data_read_summary.json",
            {
                "node_id": "data-read",
                "data_path": str(context.data_path),
                "source_format": loaded.source_format,
                "row_count": loaded.row_count,
                "column_count": loaded.column_count,
                "id_cols": list(result.contract.id_cols),
                "date_col": result.contract.date_col,
                "target_col": result.contract.target_col,
                "feature_cols": list(result.feature_cols),
                "id_unique_rate": result.id_unique_rate,
                "date_parse_rate": result.date_parse_rate,
                "node_config": str(context.config.path),
                "node_config_sha256": config_hash,
            },
        )
    confirmation = write_json(
        context.output_dir / "data_read_confirmation.json",
        {
            "node_id": "data-read",
            "status": "awaiting_user_confirmation",
            "message": "数据已读取。请查看 configs/node_configs/data-read.yaml 并确认数据路径、读取参数和字段角色（ID/日期/目标）；当前节点不执行样本诊断、数据切分或模型训练。确认后重新执行并追加 --confirm-config。",
            "config_path": str(context.config.path),
            "config_sha256": config_hash,
            "data_path": str(context.data_path),
            "summary_artifact": str(artifact),
            "display_files": [str(context.config.path)],
        },
    )
    if not args.confirm_config:
        context.progress.emit(
            "data-read",
            "waiting_confirmation",
            summary="等待用户确认 data-read.yaml",
            artifacts=[artifact, confirmation],
        )
        return {
            "status": "awaiting_user_confirmation",
            "node_id": "data-read",
            "artifact": str(artifact),
            "confirmation": str(confirmation),
            "config": str(context.config.path),
            "display_files": [str(context.config.path)],
        }

    with context.progress.track(
        "data-read-validation",
        running_summary="正在执行已确认的数据契约校验",
        success_summary="数据读取节点完成",
        artifacts=[artifact, confirmation],
    ):
        validate_contract(
            loaded.data,
            context.effective_contract,
            allow_id_duplicates=bool(
                context.config.parameters.get("allow_id_duplicates", False)
            ),
            allow_target_issues=bool(
                context.config.parameters.get("allow_target_issues", False)
            ),
        )
    approval = write_json(
        context.node_config_dir / "data-read.approval.json",
        {
            "node_id": "data-read",
            "status": "approved",
            "config_path": str(context.config.path),
            "config_sha256": config_hash,
            "data_path": str(context.data_path),
        },
    )
    context.progress.emit(
        "data-read",
        "success",
        summary="数据读取与契约校验完成",
        artifacts=[artifact, approval],
    )
    return {
        "status": "success",
        "node_id": "data-read",
        "artifact": str(artifact),
        "approval": str(approval),
    }


def main() -> int:
    result = run(build_parser().parse_args())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
