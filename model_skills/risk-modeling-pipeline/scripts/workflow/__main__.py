"""Workflow package entry point for the risk-modeling Skill."""

import sys

from .runner import main
from .definition import MAIN_NODE_IDS


_NODE_EXECUTORS = {
    "data-read": "data_read.run",
    "sample-diagnosis": "sample_diagnosis.run",
    "eda-analysis": "eda_analysis.run",
    "feature-processing": "feature_processing.run",
}


if __name__ == "__main__":
    # Independent node execution is the preferred OpenCode integration. The
    # multi-stage runner remains available when no node is specified.
    if len(sys.argv) > 1 and (sys.argv[1] in MAIN_NODE_IDS or sys.argv[1] in {"sample-diagnosis", "eda-analysis"}):
        node_id = sys.argv[1]
        module_name = _NODE_EXECUTORS.get(node_id)
        if module_name is None:
            raise SystemExit(
                f"Workflow node {node_id!r} has no standalone executor yet; "
                "use the batch runner or enable the node implementation first."
            )
        module = __import__(f"workflow.nodes.{module_name}", fromlist=["main"])
        del sys.argv[1]
        raise SystemExit(module.main())
    raise SystemExit(main())
