import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_wearing import data
from online_wearing.contracts import MEASUREMENT_FIELDS
from online_wearing.evaluation import evaluate_subset


class FakeImage:
    def save(self, path):
        Path(path).write_bytes(b"fake image bytes")


class Rows:
    def __init__(self, size):
        self.size, self.consumed = size, 0

    def __iter__(self):
        for index in range(self.size):
            self.consumed += 1
            yield dict(cloth=object(), person=object(), target=object(),
                       **{name: index + .5 for name in MEASUREMENT_FIELDS})


class FakeResult:
    image = object()

    def save(self, path, debug=True):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "result.png").write_bytes(b"fake result")


class FakePipeline:
    weights = SimpleNamespace(as_dict=lambda: {"kg": "fixed.pt"})
    sampling_steps, repaint_repeats, sci_precision, device = 50, 10, "float32", "cpu"

    def __init__(self, fail_id=None):
        self.calls, self.fail_id = [], fail_id

    def __call__(self, cloth, person, garment_type, *, seed):
        self.calls.append((cloth, person, garment_type, seed))
        if Path(cloth).parent.name == self.fail_id:
            raise ValueError("No person detected")
        return FakeResult()


class DataEvaluationTest(unittest.TestCase):
    def export(self, root, size=2, limit=2):
        rows = Rows(size)
        with patch.object(data, "_image", return_value=FakeImage()):
            manifest = data.export_rows(rows, root, limit=limit, default_garment_type="long sleeve top",
                                        garment_types={"000001": "vest"})
        return rows, manifest

    def metrics(self, image, target):
        self.assertIs(image, FakeResult.image)
        self.assertEqual(Path(target).name, "target.png")
        return dict(mae=.1, psnr_db=20., ssim=.8)

    def test_export_stops_without_reading_extra_row_and_preserves_cm(self):
        with tempfile.TemporaryDirectory() as root:
            rows, manifest = self.export(root, size=2000)
            records = data.read_manifest(manifest)
            self.assertEqual(rows.consumed, 2)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[1]["measurements_cm"]["body_bust"], 1.5)
            self.assertEqual(records[1]["garment_type"], "vest")
            self.assertEqual(records[0]["garment_type_source"], "explicit_default")
            self.assertEqual(records[1]["garment_type_source"], "override")
            self.assertFalse(Path(records[0]["cloth"]).is_absolute())

    def test_export_refuses_more_than_1000(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                data.export_rows([], root, limit=1001, default_garment_type="vest")

    def test_short_stream_does_not_publish_complete_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "stream ended"):
                self.export(root, size=1, limit=2)
            self.assertFalse((Path(root) / "manifest.jsonl").exists())

    def test_schema_typo_fails_instead_of_using_target(self):
        row = {"image": object(), "person": object(), "target": object()}
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "cloth"):
                data.export_rows([row], root, limit=1, default_garment_type="vest")

    def test_evaluation_target_is_only_given_to_metric_and_successes_resume(self):
        with tempfile.TemporaryDirectory() as root:
            _, manifest = self.export(Path(root) / "data")
            pipeline = FakePipeline()
            output = Path(root) / "run"
            summary = evaluate_subset(pipeline, manifest, output, metric_fn=self.metrics)
            self.assertEqual(summary["succeeded"], 2)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual([call[0].name for call in pipeline.calls], ["cloth.png"] * 2)
            self.assertEqual([call[1].name for call in pipeline.calls], ["person.png"] * 2)
            self.assertEqual([call[3] for call in pipeline.calls], [42, 43])
            evaluate_subset(pipeline, manifest, output, metric_fn=self.metrics)
            self.assertEqual(len(pipeline.calls), 2)

    def test_failures_reported_and_retried(self):
        with tempfile.TemporaryDirectory() as root:
            _, manifest = self.export(Path(root) / "data")
            output = Path(root) / "run"
            pipeline = FakePipeline(fail_id="000001")
            summary = evaluate_subset(pipeline, manifest, output, metric_fn=self.metrics)
            self.assertEqual((summary["succeeded"], summary["failed"]), (1, 1))
            self.assertEqual(summary["errors"], {"ValueError": 1})
            pipeline.fail_id = None
            summary = evaluate_subset(pipeline, manifest, output, metric_fn=self.metrics)
            self.assertEqual(len(pipeline.calls), 3)
            self.assertEqual((summary["succeeded"], summary["failed"]), (2, 0))

    def test_resume_refuses_changed_sampling_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            _, manifest = self.export(Path(root) / "data")
            output, pipeline = Path(root) / "run", FakePipeline()
            evaluate_subset(pipeline, manifest, output, metric_fn=self.metrics)
            pipeline.sampling_steps = 10
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                evaluate_subset(pipeline, manifest, output, metric_fn=self.metrics)

    def test_fail_fast_records_error_before_raising(self):
        with tempfile.TemporaryDirectory() as root:
            _, manifest = self.export(Path(root) / "data")
            output = Path(root) / "run"
            with self.assertRaisesRegex(ValueError, "No person"):
                evaluate_subset(FakePipeline("000000"), manifest, output, fail_fast=True, metric_fn=self.metrics)
            self.assertEqual(data.read_manifest(output / "metrics.jsonl")[0]["status"], "error")


if __name__ == "__main__":
    unittest.main()
