"""Integration checks for preparation findings and hash-bound approval."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ai.approval import (
    ApprovalError,
    canonical_sha256,
    create_approval_manifest,
    file_sha256,
    validate_approval_manifest,
)
from ai.pipeline import prepare_modeling_request
from data.contract import load_contract, validate_contract
from data.loader import load_table
from modeling.config import load_model_config
from preprocessing.sample_config import load_sample_config


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class AiWorkflowIntegrationTest(unittest.TestCase):
    def test_prepare_detects_current_data_blockers(self) -> None:
        data_path = PROJECT_ROOT / "data.csv"
        contract_path = PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "data_contract.yaml"
        config_path = PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "model_config.yaml"
        sample_config_path = PROJECT_ROOT / "risk-modeling-pipeline" / "scripts" / "test_fixtures" / "sample_config.yaml"
        data = load_table(data_path).data
        contract = validate_contract(data, load_contract(contract_path))
        config = load_model_config(config_path)
        sample_config = load_sample_config(sample_config_path)
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = prepare_modeling_request(
                data,
                contract,
                planned_mode="all",
                data_path=data_path,
                contract_path=contract_path,
                model_config=config,
                model_config_path=config_path,
                sample_config=sample_config,
                sample_config_path=sample_config_path,
                output_dir=temporary_directory,
            )
            request = json.loads(
                result.confirmation_request_path.read_text(encoding="utf-8")
            )
            self.assertTrue(Path(request["review_artifacts"]["feature_proposal"]).is_file())
            self.assertTrue(Path(request["review_artifacts"]["feature_preprocessing"]).is_file())
            self.assertTrue(Path(request["review_artifacts"]["feature_selection"]).is_file())
            self.assertTrue(Path(request["review_artifacts"]["model_config_proposal"]).is_file())
        self.assertEqual(request["status"], "awaiting_user_confirmation")
        self.assertIn("LATEST_MONTH_INCOMPLETE", request["blocker_codes"])
        self.assertIn("OOT_BAD_COUNT_CRITICAL", request["blocker_codes"])
        self.assertIn(
            "CLASS_IMBALANCE_WARNING",
            {item["code"] for item in request["sample_findings"]},
        )
        self.assertEqual(
            request["sample_treatment"]["class_imbalance_action"], "class_weight"
        )
        self.assertIn("sample_diagnostics", request["review_artifacts"])
        self.assertIn("feature_proposal", request["review_artifacts"])
        self.assertIn("feature_preprocessing", request["review_artifacts"])
        self.assertIn("feature_selection", request["review_artifacts"])
        self.assertIn("model_config_proposal", request["review_artifacts"])
        self.assertEqual(request["feature_preview"]["fit_scope"], "train_only")
        self.assertEqual(request["feature_preview"]["retained_feature_count"], 19)
        self.assertEqual(request["feature_policy"]["fit_scope"], "train_only")
        self.assertEqual(
            request["feature_policy"]["preprocessing"]["categorical"]["strategy"],
            "lightgbm_native",
        )
        self.assertEqual(
            request["feature_policy"]["selection"]["stability_action"],
            "review",
        )

    def test_approval_requires_blocker_acknowledgement_and_current_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "data.csv"
            contract_path = root / "contract.yaml"
            config_path = root / "model.yaml"
            sample_config_path = root / "sample.yaml"
            data_path.write_text("x\n1\n", encoding="utf-8")
            contract_path.write_text("schema: {}\n", encoding="utf-8")
            config_path.write_text("model: {}\n", encoding="utf-8")
            sample_config_path.write_text("treatment: {}\n", encoding="utf-8")
            request_path = root / "confirmation_request.json"
            request_body = {
                "status": "awaiting_user_confirmation",
                "planned_mode": "all",
                "blocker_codes": ["BLOCKER_TEST"],
                "artifacts": {
                    "data": {"path": str(data_path), "sha256": file_sha256(data_path)},
                    "data_contract": {
                        "path": str(contract_path),
                        "sha256": file_sha256(contract_path),
                    },
                    "model_config": {
                        "path": str(config_path),
                        "sha256": file_sha256(config_path),
                    },
                    "sample_config": {
                        "path": str(sample_config_path),
                        "sha256": file_sha256(sample_config_path),
                    },
                },
            }
            request_path.write_text(
                json.dumps(
                    {**request_body, "request_id": canonical_sha256(request_body)}
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ApprovalError):
                create_approval_manifest(request_path, confirmed_by="test-user")
            manifest_path = create_approval_manifest(
                request_path,
                confirmed_by="test-user",
                acknowledged_findings={"BLOCKER_TEST"},
            )
            tampered = json.loads(request_path.read_text(encoding="utf-8"))
            tampered["blocker_codes"] = []
            request_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(ApprovalError):
                create_approval_manifest(request_path, confirmed_by="test-user")
            validate_approval_manifest(
                manifest_path,
                planned_mode="all",
                data_path=data_path,
                contract_path=contract_path,
                sample_config_path=sample_config_path,
                model_config_path=config_path,
            )
            config_path.write_text("model: changed\n", encoding="utf-8")
            with self.assertRaises(ApprovalError):
                validate_approval_manifest(
                    manifest_path,
                    planned_mode="all",
                    data_path=data_path,
                    contract_path=contract_path,
                    sample_config_path=sample_config_path,
                    model_config_path=config_path,
                )


if __name__ == "__main__":
    unittest.main()
