"""Workflow package entry point for the risk-modeling Skill."""

import sys

from .runner import main
from .definition import MAIN_NODE_IDS


_NODE_EXECUTORS = {
    "data-read": "data_read.run",
    "sample-diagnosis": "sample_diagnosis.run",
    "eda-analysis": "eda_analysis.run",
    "feature-processing": "feature_processing.run",
    "model-config": "model_config.run",
    "training-tuning": "training_tuning.run",
    # Compatibility alias: LLM refinement is implemented inside the
    # training-tuning executor, not as a separate user-facing node.
    "llm-tuning": "training_tuning.run",
}

# Keep the historical short name accepted by older prompts and shell snippets,
# while always dispatching the canonical user-facing node internally.  This is
# deliberately a small compatibility map; new integrations should use the
# canonical ``eda-analysis`` ID.
_NODE_ALIASES = {
    "eda": "eda-analysis",
}

# Only data-read exposes a user-editable YAML confirmation gate. Later nodes
# run with their current YAML/defaults, then expose a result confirmation gate;
# textual user changes are applied to the current node YAML and rerun by host.
_CONFIRMABLE_NODES = {
    "data-read",
    "eda-analysis",
    "sample-diagnosis",
    "feature-processing",
    "model-config",
}


if __name__ == "__main__":
    # Deterministic confirmation entry point:
    #   python -m workflow confirm <node-id> [node options]
    # The command is translated to the node's confirmation flag: data-read
    # confirms YAML, while later nodes confirm the already-generated result.
    if len(sys.argv) > 2 and sys.argv[1] == "confirm":
        requested_node_id = sys.argv[2]
        node_id = _NODE_ALIASES.get(requested_node_id, requested_node_id)
        if node_id not in _CONFIRMABLE_NODES:
            valid = ", ".join(sorted(_CONFIRMABLE_NODES))
            raise SystemExit(f"Node {requested_node_id!r} is not confirmable. Valid nodes: {valid}")
        module_name = _NODE_EXECUTORS[node_id]
        del sys.argv[1:3]
        confirmation_flag = "--confirm-config" if node_id == "data-read" else "--confirm-result"
        if confirmation_flag not in sys.argv and "--confirm-config" not in sys.argv and "--confirm-result" not in sys.argv:
            sys.argv.append(confirmation_flag)
        module = __import__(f"workflow.nodes.{module_name}", fromlist=["main"])
        raise SystemExit(module.main())

    # Independent node execution is the preferred OpenCode integration. The
    # multi-stage runner remains available when no node is specified.
    requested_node_id = sys.argv[1] if len(sys.argv) > 1 else ""
    node_id = _NODE_ALIASES.get(requested_node_id, requested_node_id)
    if node_id in MAIN_NODE_IDS or node_id in {"sample-diagnosis", "eda-analysis", "llm-tuning"}:
        module_name = _NODE_EXECUTORS.get(node_id)
        if module_name is None:
            raise SystemExit(
                f"Workflow node {node_id!r} has no standalone executor yet; "
                "use the batch runner or enable the node implementation first."
            )
        module = __import__(f"workflow.nodes.{module_name}", fromlist=["main"])
        del sys.argv[1]
        if node_id == "llm-tuning" and "--llm-only" not in sys.argv:
            sys.argv.append("--llm-only")
        raise SystemExit(module.main())
    raise SystemExit(main())
