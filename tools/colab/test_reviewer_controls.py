"""Dependency-free checks for source reuse and immutable transfer manifests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("reviewer_controls", HERE / "reviewer_controls.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReviewerControlTests(unittest.TestCase):
    def test_notebook_adaptations_compile_and_leave_source_unchanged(self):
        path = HERE.parents[1] / module.NOTEBOOK
        before = path.read_bytes()
        notebook = json.loads(before)
        for index in (*module.DATA_CELLS, 25, 26, 27):
            compile(module.adapted_source(notebook, index), str(index), "exec")
        compile(module.CONTROL_SOURCE, "extension", "exec")
        self.assertEqual(path.read_bytes(), before)

    def test_source_drift_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Notebook source drift"):
            module.replace_once("changed original implementation", "old implementation", "replacement")

    def test_checkpoint_chunks_reassemble_and_remain_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            source = output / "checkpoints" / "run_latest.pt"
            source.parent.mkdir()
            original = b"checkpoint payload 1 with optimizer, scaler, RNG state"
            source.write_bytes(original)
            publisher = module.ArtifactPublisher(output)
            publisher.CHUNK_BYTES = 7
            publisher.checkpoint(source, "run", 1, "latest")
            source.write_bytes(b"next checkpoint changed the mutable latest file")
            manifest = json.loads((output / "artifact_manifest.json").read_text())
            self.assertFalse(manifest["complete"])
            artifact = manifest["artifacts"][0]
            self.assertFalse((output / artifact["path"]).exists())
            result = b"".join((output / item["path"]).read_bytes() for item in artifact["chunks"])
            self.assertEqual(result, original)
            self.assertEqual(hashlib.sha256(result).hexdigest(), artifact["sha256"])
            for item in artifact["chunks"]:
                data = (output / item["path"]).read_bytes()
                self.assertEqual(len(data), item["size_bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            publisher.complete = True
            publisher.flush()
            self.assertTrue(json.loads((output / "artifact_manifest.json").read_text())["complete"])

    def test_result_snapshots_change_address_with_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            metrics = output / "metrics"
            metrics.mkdir()
            source = metrics / "history.json"
            publisher = module.ArtifactPublisher(output)
            source.write_text('[{"epoch": 1}]')
            publisher.compact_results()
            first = dict(publisher.artifacts)
            source.write_text('[{"epoch": 1}, {"epoch": 2}]')
            publisher.compact_results()
            self.assertEqual(len(publisher.artifacts), 2)
            for path, artifact in first.items():
                self.assertEqual(hashlib.sha256((output / path).read_bytes()).hexdigest(), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
