import unittest

from scripts.validate_packed_nf4_split_manifest import validate_rows


def row(artifact, run, role, split, block, source="model.layer.q", tensor="q_proj"):
    return {
        "artifact_sha256": artifact,
        "run_id": run,
        "source_id": source,
        "tensor_key": tensor,
        "role": role,
        "split": split,
        "block_index": str(block),
    }


class ValidatePackedNF4SplitManifestTests(unittest.TestCase):
    def valid_rows(self):
        return [
            row("clean-hash", "run-a", "clean", "train", 0),
            row("embedded-hash", "run-a", "embedded", "train", 0),
            row("clean-hash", "run-a", "clean", "train", 1),
            row("embedded-hash", "run-a", "embedded", "train", 1),
            row("clean-hash-b", "run-b", "clean", "test", 0),
            row("embedded-hash-b", "run-b", "embedded", "test", 0),
        ]

    def test_accepts_paired_artifacts_confined_to_one_split(self):
        result = validate_rows(self.valid_rows())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["unique_runs"], 2)

    def test_rejects_artifact_across_splits(self):
        rows = self.valid_rows()
        rows.append(row("clean-hash", "run-a", "clean", "test", 2))
        result = validate_rows(rows)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("artifact clean-hash appears in multiple splits" in e
                            for e in result["errors"]))

    def test_rejects_run_across_splits(self):
        rows = self.valid_rows()
        rows.append(row("another-hash", "run-a", "embedded", "test", 2))
        result = validate_rows(rows)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("run run-a appears in multiple splits" in e
                            for e in result["errors"]))

    def test_rejects_unpaired_block_indices(self):
        rows = self.valid_rows()
        rows = [r for r in rows if not (
            r["artifact_sha256"] == "embedded-hash" and r["block_index"] == "1"
        )]
        result = validate_rows(rows)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("block indices differ" in e for e in result["errors"]))


if __name__ == "__main__":
    unittest.main()
