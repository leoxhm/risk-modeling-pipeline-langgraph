"""Standalone data-read workflow node."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json

from ..common import NodeContext, approval_path, write_json, write_node_summary


def _display_value(value: object, *, empty: str = "未设置") -> str:
    """Render a YAML/contract value safely inside a Markdown table cell."""
    if value is None or value == "":
        return empty
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(item) for item in value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _build_summary_markdown(context: NodeContext, loaded, result) -> str:
    """Build the stable, compact table returned to the host UI.

    The table is generated from the resolved YAML roles and the actual loaded
    data, so the model/host does not need to infer row counts or field roles
    from logs.  Detailed JSON remains available as a machine-readable
    artifact.
    """
    contract = result.contract
    parameters = context.config.parameters
    features = list(result.feature_cols)
    feature_preview = ", ".join(f"`{name}`" for name in features[:10])
    if len(features) > 10:
        feature_preview += f" 等（共 {len(features)} 个）"
    elif features:
        feature_preview += f"（共 {len(features)} 个）"
    else:
        feature_preview = "—"
    excluded = ", ".join(f"`{name}`" for name in contract.exclude_cols) or "—"
    labels = (
        f"好 = `{_display_value(contract.good_label)}`；"
        f"坏 = `{_display_value(contract.bad_label)}`"
    )
    rows = [
        ("数据文件", f"`{_display_value(context.data_path.name)}`"),
        ("文件格式", _display_value(loaded.source_format).upper()),
        ("样本量", f"{loaded.row_count:,} 行 × {loaded.column_count} 列"),
        (
            "ID 字段",
            f"{', '.join(f'`{name}`' for name in contract.id_cols)}"
            f"（唯一率 {result.id_unique_rate:.2%}）",
        ),
        (
            "日期字段",
            f"`{_display_value(contract.date_col)}`（解析率 {result.date_parse_rate:.2%}）",
        ),
        ("目标字段（Y）", f"`{_display_value(contract.target_col)}`"),
        ("好坏标签映射", labels),
        ("特征字段", feature_preview),
        ("排除字段", excluded),
        (
            "读取参数",
            "; ".join(
                [
                    f"encoding={_display_value(parameters.get('encoding'))}",
                    f"sheet_name={_display_value(parameters.get('sheet_name'))}",
                    f"infer_schema_length={_display_value(parameters.get('infer_schema_length'))}",
                ]
            ),
        ),
        (
            "校验容忍开关",
            f"allow_id_duplicates={_display_value(parameters.get('allow_id_duplicates'))}; "
            f"allow_target_issues={_display_value(parameters.get('allow_target_issues'))}",
        ),
    ]
    table = ["## data-read 结果", "", "| 项目 | 值 |", "|---|---|"]
    table.extend(f"| {name} | {value} |" for name, value in rows)
    table.extend(
        [
            "",
            "以上字段角色和读取参数来自已解析的 YAML/数据契约，统计值来自本次实际读取。",
            "请确认表格内容后再继续；如需调整，请直接用文字说明字段或读取策略。",
        ]
    )
    return "\n".join(table)


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
        summary_markdown = _build_summary_markdown(context, loaded, result)
        summary_markdown_path = context.output_dir / "data_read_summary.md"
        summary_markdown_path.write_text(summary_markdown + "\n", encoding="utf-8")
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
                "summary": summary_markdown,
            },
        )
    confirmation = write_json(
        context.output_dir / "data_read_confirmation.json",
        {
            "node_id": "data-read",
            "status": "awaiting_user_confirmation",
            "message": "数据已读取。请确认上方 Markdown 表格中的数据文件、读取参数和字段角色（ID/日期/目标）；如需调整请直接用文字说明。当前节点不执行样本诊断、数据切分或模型训练。确认后继续。",
            "config_path": str(context.config.path),
            "config_sha256": config_hash,
            "data_path": str(context.data_path),
            "summary_artifact": str(artifact),
            "summary_markdown": str(summary_markdown_path),
            "summary": summary_markdown,
            "display_files": [str(summary_markdown_path)],
        },
    )
    if not args.confirm_config:
        # Replace any previous approval with a pending marker so downstream
        # nodes cannot reuse an approval after the read configuration changes.
        pending_approval = write_json(
            approval_path(context, "data-read", for_write=True),
            {
                "node_id": "data-read",
                "status": "awaiting_user_confirmation",
                "config_path": str(context.config.path),
                "config_sha256": config_hash,
                "data_path": str(context.data_path),
            },
        )
        context.progress.emit(
            "data-read",
            "waiting_confirmation",
            summary=summary_markdown,
            artifacts=[artifact, summary_markdown_path, confirmation, pending_approval],
        )
        return {
            "status": "awaiting_user_confirmation",
            "node_id": "data-read",
            "artifact": str(artifact),
            "confirmation": str(confirmation),
            "approval": str(pending_approval),
            "config": str(context.config.path),
            "summary_markdown": str(summary_markdown_path),
            "display_files": [str(summary_markdown_path)],
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
        approval_path(context, "data-read", for_write=True),
        {
            "node_id": "data-read",
            "status": "approved",
            "config_path": str(context.config.path),
            "config_sha256": config_hash,
            "data_path": str(context.data_path),
        },
    )
    node_summary = write_node_summary(
        context.output_dir,
        "data-read",
        summary_markdown,
        {
            "data_path": str(context.data_path),
            "row_count": loaded.row_count,
            "column_count": loaded.column_count,
            "feature_count": len(result.feature_cols),
            "id_unique_rate": result.id_unique_rate,
            "date_parse_rate": result.date_parse_rate,
            "summary_markdown": str(summary_markdown_path),
            "recommendations": [
                "确认 ID、日期、标签字段及标签含义，并复核是否存在不应入模的泄漏字段。"
            ],
        },
    )
    context.progress.emit(
        "data-read",
        "success",
        summary=summary_markdown,
        artifacts=[artifact, summary_markdown_path, approval, node_summary],
    )
    return {
        "status": "success",
        "node_id": "data-read",
        "artifact": str(artifact),
        "summary": summary_markdown,
        "summary_markdown": str(summary_markdown_path),
        "approval": str(approval),
        "node_summary": str(node_summary),
    }


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
