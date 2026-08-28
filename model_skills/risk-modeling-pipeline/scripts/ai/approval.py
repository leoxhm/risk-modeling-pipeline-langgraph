"""Hash-bound human approval manifests for modeling runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class ApprovalError(ValueError):
    """Raised when a run is attempted without a valid current approval."""


def file_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApprovalError(f"Cannot read approval JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ApprovalError(f"Approval JSON must contain an object: {source}")
    return value


def create_approval_manifest(
    confirmation_request_path: str | Path,
    *,
    confirmed_by: str,
    acknowledged_findings: set[str] | None = None,
    confirmed_nodes: set[str] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Create an approval only after the caller has obtained explicit user consent."""

    if not confirmed_by.strip():
        raise ApprovalError("confirmed_by must identify the user who approved the run")
    request_path = Path(confirmation_request_path).expanduser().resolve()
    request = _load_json(request_path)
    if request.get("status") != "awaiting_user_confirmation":
        raise ApprovalError("Confirmation request is not awaiting user confirmation")
    request_body = {key: value for key, value in request.items() if key != "request_id"}
    if request.get("request_id") != canonical_sha256(request_body):
        raise ApprovalError(
            "Confirmation request changed after preparation; run prepare again"
        )
    acknowledged = acknowledged_findings or set()
    blockers = set(request.get("blocker_codes", []))
    missing_acknowledgements = sorted(blockers - acknowledged)
    if missing_acknowledgements:
        raise ApprovalError(
            "Blocker findings require explicit acknowledgement: "
            + ", ".join(missing_acknowledgements)
        )
    required_nodes = set(request.get("selected_steps") or ())
    missing_nodes = sorted(required_nodes - (confirmed_nodes or set()))
    if missing_nodes:
        raise ApprovalError(
            "Each selected node requires explicit config confirmation: "
            + ", ".join(missing_nodes)
        )
    manifest = {
        "schema_version": 1,
        "status": "approved",
        "request_id": request["request_id"],
        "planned_mode": request["planned_mode"],
        "selected_steps": request.get("selected_steps"),
        "confirmed_nodes": sorted(confirmed_nodes or required_nodes),
        "confirmed_by": confirmed_by.strip(),
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
        "acknowledged_findings": sorted(acknowledged),
        "artifacts": request["artifacts"],
    }
    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else request_path.with_name("approval_manifest.json")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def validate_approval_manifest(
    manifest_path: str | Path,
    *,
    planned_mode: str,
    data_path: str | Path,
    contract_path: str | Path,
    sample_config_path: str | Path,
    model_config_path: str | Path | None,
    require_model_config: bool = True,
    selected_steps: Sequence[str] | None = None,
    node_config_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Reject stale, mismatched, or scope-incompatible approvals."""

    manifest = _load_json(manifest_path)
    if manifest.get("status") != "approved":
        raise ApprovalError("Approval manifest status is not approved")
    if manifest.get("planned_mode") != planned_mode:
        raise ApprovalError(
            f"Approval mode is {manifest.get('planned_mode')!r}, requested mode is {planned_mode!r}"
        )
    if selected_steps is not None:
        approved_steps = tuple(manifest.get("selected_steps") or ())
        if approved_steps != tuple(selected_steps):
            raise ApprovalError(
                "Approval selected workflow nodes do not match the requested run"
            )
    expected_paths = {
        "data": Path(data_path).expanduser().resolve(),
        "data_contract": Path(contract_path).expanduser().resolve(),
        "sample_config": Path(sample_config_path).expanduser().resolve(),
    }
    if planned_mode in {"model", "all"} and require_model_config:
        if model_config_path is None:
            raise ApprovalError("Model runs require a model configuration path")
        expected_paths["model_config"] = Path(model_config_path).expanduser().resolve()
    approved_artifacts = manifest.get("artifacts", {})
    for name, path in expected_paths.items():
        approved = approved_artifacts.get(name)
        if not approved:
            raise ApprovalError(f"Approval manifest does not cover {name}")
        if Path(approved["path"]).resolve() != path:
            raise ApprovalError(
                f"Approved {name} path does not match the requested run"
            )
        current_hash = file_sha256(path)
        if approved.get("sha256") != current_hash:
            raise ApprovalError(f"Approved {name} changed after user confirmation")
    if node_config_paths:
        approved_nodes = approved_artifacts.get("node_configs", {})
        for node_id, node_path in node_config_paths.items():
            path = Path(node_path).expanduser().resolve()
            approved = approved_nodes.get(node_id)
            if not approved:
                raise ApprovalError(f"Approval manifest does not cover node config {node_id}")
            if Path(approved["path"]).resolve() != path:
                raise ApprovalError(f"Approved node config path does not match: {node_id}")
            if approved.get("sha256") != file_sha256(path):
                raise ApprovalError(f"Node config changed after user confirmation: {node_id}")
    return manifest
