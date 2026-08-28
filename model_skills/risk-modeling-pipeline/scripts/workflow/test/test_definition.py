"""Tests for the stable user-facing workflow contract."""

from __future__ import annotations

import unittest

from workflow import MAIN_NODE_IDS, WORKFLOW_NODES, main_node_for


class WorkflowDefinitionTest(unittest.TestCase):
    def test_exposes_eight_main_nodes_in_execution_order(self) -> None:
        self.assertEqual(
            [node.id for node in WORKFLOW_NODES],
            [
                "data-read",
                "eda-analysis",
                "sample-diagnosis",
                "feature-processing",
                "model-config",
                "training-tuning",
                "model-review",
                "report-delivery",
            ],
        )
        self.assertEqual(len(MAIN_NODE_IDS), 8)

    def test_groups_internal_events_without_changing_the_audit_ids(self) -> None:
        self.assertEqual(main_node_for("loader"), "data-read")
        self.assertEqual(main_node_for("feature-selection"), "feature-processing")
        self.assertEqual(main_node_for("model-report"), "report-delivery")
        self.assertEqual(main_node_for("training-tuning"), "training-tuning")


if __name__ == "__main__":
    unittest.main()
