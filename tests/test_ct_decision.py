"""Pruebas del controlador ligero de Continuous Training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ct_decision import evaluate


EVIDENCE = Path("docs/week9-evidence")


class TestCTDecision(unittest.TestCase):

    def test_real_evidence_keeps_v1(self):
        result = evaluate(EVIDENCE)
        self.assertTrue(result["drift"]["detected"])
        self.assertEqual(result["drift"]["affected_columns"], 14)
        self.assertEqual(result["drift"]["total_columns"], 19)
        self.assertEqual(
            result["retraining"]["decision"],
            "REVISAR_REENTRENAMIENTO",
        )
        self.assertEqual(result["promotion"]["decision"], "CONSERVAR_V1")
        self.assertFalse(result["promotion"]["automatic_execution"])

    def test_no_drift_continues_monitoring(self):
        original = json.loads((EVIDENCE / "evidently.json").read_text())
        for metric in original["metrics"]:
            if metric["metric"] == "DatasetDriftMetric":
                metric["result"]["dataset_drift"] = False
                metric["result"]["number_of_drifted_columns"] = 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidently.json").write_text(json.dumps(original))
            ab_dir = root / "ab-v1-v3-20261115"
            ab_dir.mkdir()
            (ab_dir / "ab_results.json").write_bytes(
                (EVIDENCE / "ab-v1-v3-20261115" / "ab_results.json").read_bytes()
            )
            result = evaluate(root)

        self.assertEqual(
            result["retraining"]["decision"],
            "CONTINUAR_MONITOREO",
        )
        self.assertFalse(result["retraining"]["review_required"])

    def test_invalid_intervals_fail_closed(self):
        original = json.loads(
            (EVIDENCE / "ab-v1-v3-20261115" / "ab_results.json").read_text()
        )
        original["paired_bootstrap_95pct_intervals"]["mae"]["low"] = 10

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidently.json").write_bytes(
                (EVIDENCE / "evidently.json").read_bytes()
            )
            ab_dir = root / "ab-v1-v3-20261115"
            ab_dir.mkdir()
            (ab_dir / "ab_results.json").write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                evaluate(root)


if __name__ == "__main__":
    unittest.main()


def test_favorable_intervals_do_not_override_limitations():
    """Intervalos favorables no bastan para promover un modelo."""
    import copy
    from unittest.mock import patch

    from scripts.ct_decision import evaluate, load_json

    original_load = load_json

    def simulated_load(path):
        data = copy.deepcopy(original_load(path))
        if path.name == "ab_results.json":
            for metric in ("mae", "rmse", "wape"):
                interval = data["paired_bootstrap_95pct_intervals"][metric]
                if metric == "wape":
                    interval = interval["proportion"]
                interval["low"] = -0.5
                interval["high"] = -0.1
            data["reference_comparison"]["all_within_tolerance"] = True
        return data

    with patch("scripts.ct_decision.load_json", side_effect=simulated_load):
        result = evaluate(EVIDENCE)

    assert all(
        item["improvement_supported"]
        for item in result["ab_intervals"].values()
    )
    assert result["promotion"]["decision"] == "CONSERVAR_V1"
    assert result["promotion"]["automatic_execution"] is False
