"""Shared context helpers for independently executed workflow nodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from data.contract import DataContract, apply_role_overrides, load_contract
from data.loader import DataLoadResult, discover_data_file, load_table
from progress import ProgressReporter
from workflow.node_config import NodeConfig, ensure_node_configs


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_engine_assets(args: Any) -> Path:
    """Resolve the installed Skill assets directory for workspace bootstrap."""
    if getattr(args, "engine_root", None):
        root = Path(args.engine_root).expanduser().resolve()
    else:
        # <engine-root>/risk-modeling-pipeline/scripts/workflow/nodes/common.py
        root = Path(__file__).resolve().parents[4]
    candidates = (root / "risk-modeling-pipeline" / "assets", root / "assets")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot locate Skill assets under engine root: {root}")


def ensure_workspace_configs(
    project_root: Path, assets_dir: Path, required_nodes: set[str] | None = None
) -> None:
    """Copy only templates needed by the selected node and its prerequisites.

    ``data_contract.yaml`` is intentionally shared by every node. Node YAMLs
    are copied on demand; model_config.yaml is not created during data/EDA
    work.
    """
    config_dir = project_root / "configs"
    node_config_dir = config_dir / "node_configs"
    node_config_dir.mkdir(parents=True, exist_ok=True)
    nodes = set(required_nodes or set())
    templates = {"data_contract.template.yaml": config_dir / "data_contract.yaml"}
    node_templates = {
        "data-read": ("data_read.template.yaml", "data-read.yaml"),
        "eda-analysis": ("eda_analysis.template.yaml", "eda-analysis.yaml"),
        "sample-diagnosis": ("sample_diagnosis.template.yaml", "sample-diagnosis.yaml"),
        "feature-processing": ("feature_processing.template.yaml", "feature-processing.yaml"),
        "model-config": ("model_config_node.template.yaml", "model-config.yaml"),
        "training-tuning": ("training_tuning.template.yaml", "training-tuning.yaml"),
        "model-review": ("model_review.template.yaml", "model-review.yaml"),
        "report-delivery": ("report_delivery.template.yaml", "report-delivery.yaml"),
    }
    # Every executable downstream node needs data-read. EDA-dependent nodes
    # also need the EDA configuration present for their prerequisite gate.
    if nodes:
        nodes.add("data-read")
    if nodes & {"sample-diagnosis", "feature-processing"}:
        nodes.add("eda-analysis")
    if nodes & {"model-config", "training-tuning", "model-review", "report-delivery"}:
        templates["model_config.template.yaml"] = config_dir / "model_config.yaml"
    for node_id in nodes:
        if node_id in node_templates:
            template_name, target_name = node_templates[node_id]
            templates[template_name] = node_config_dir / target_name
    for template_name, target in templates.items():
        source = assets_dir / template_name
        if not target.exists() and source.is_file():
            shutil.copy2(source, target)


@dataclass(frozen=True)
class NodeContext:
    project_root: Path
    data_path: Path
    contract_path: Path
    node_config_dir: Path
    output_dir: Path
    config: NodeConfig
    data_read_config: NodeConfig
    contract: DataContract
    effective_contract: DataContract
    progress: ProgressReporter

    @classmethod
    def from_args(cls, args: Any, node_id: str) -> "NodeContext":
        project_root = Path(args.project_root).expanduser().resolve()
        assets_dir = resolve_engine_assets(args)
        required_nodes = {node_id}
        if node_id in {"eda-analysis", "sample-diagnosis", "feature-processing"}:
            required_nodes.add("data-read")
        ensure_workspace_configs(project_root, assets_dir, required_nodes)
        contract_path = resolve_path(
            args.data_contract or "configs/data_contract.yaml", project_root
        )
        contract = load_contract(contract_path)
        # Resolve the source deterministically. Explicit CLI path wins, then
        # the contract path, and only then a unique root-level data file.
        if args.data:
            data_path = resolve_path(args.data, project_root)
        elif contract.input_path:
            data_path = resolve_path(contract.input_path, contract_path.parent)
        else:
            data_path = discover_data_file(project_root)
        node_config_dir = resolve_path(
            args.node_config_dir or "configs/node_configs", project_root
        )
        configs = ensure_node_configs(
            node_config_dir,
            ["data-read"] if node_id == "data-read" else ["data-read", node_id],
            template_dir=assets_dir,
        )
        config = configs[node_id]
        data_read_config = configs["data-read"]
        output_dir = resolve_path(
            args.output_dir or f"outputs/{node_id}", project_root
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        effective_contract = apply_role_overrides(
            contract, data_read_config.parameters
        )
        progress = ProgressReporter(output_dir, run_id=getattr(args, "run_id", None))
        return cls(
            project_root,
            data_path,
            contract_path,
            node_config_dir,
            output_dir,
            config,
            data_read_config,
            contract,
            effective_contract,
            progress,
        )

    def reload_data(self) -> DataLoadResult:
        """Reload the source using this node's current YAML parameters."""
        parameters = self.data_read_config.parameters
        options: dict[str, Any] = {}
        if self.data_path.suffix.lower() == ".csv":
            if parameters.get("encoding") is not None:
                options["encoding"] = parameters["encoding"]
            options["infer_schema_length"] = parameters.get(
                "infer_schema_length", 10000
            )
        elif self.data_path.suffix.lower() in {".xls", ".xlsx"}:
            options["sheet_name"] = parameters.get("sheet_name", 0)
        return load_table(self.data_path, **options)


def write_json(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
