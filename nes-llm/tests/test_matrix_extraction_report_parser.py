"""Regression tests for extraction-report parsing in the matrix runner."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_seven_method_long_corpus_matrix import parse_json_report


class ExtractionReportParsingTests(unittest.TestCase):
    def test_parses_json_after_device_banner(self):
        report = {"schema": "nes.seven_method_residual_extract.v1",
                  "ber": 0.0, "exact_match": True}
        stdout = "Using device: mps\n" + json.dumps(report, indent=2)
        with tempfile.TemporaryDirectory() as tmp:
            result = parse_json_report(stdout, Path(tmp) / "missing.json")
        self.assertEqual(result, report)

    def test_reads_report_file_when_stdout_has_no_json(self):
        report = {"status": "EXTRACT_EXCEPTION", "error_type": "ValueError",
                  "error": "test failure"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extract.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(parse_json_report("traceback only", path), report)

    def test_empty_output_and_missing_report_is_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(parse_json_report("not json", Path(tmp) / "missing.json"), {})


if __name__ == "__main__":
    unittest.main()
