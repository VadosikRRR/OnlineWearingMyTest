from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_wearing import data
from online_wearing.contracts import MEASUREMENT_FIELDS
from online_wearing.runtime import assert_runtime_matches_installed, StaleNotebookPackagesError

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None


def snapshot(installed, loaded):
    return {"packages": {"Pillow": {"installed": installed, "loaded": loaded}}}


class RuntimeVersionTest(unittest.TestCase):
    def test_reported_pillow_13_in_memory_12_on_disk_requires_restart(self):
        info = snapshot("12.3.0", {"PIL": "13.0.0", "PIL.Image": "13.0.0", "PIL._imaging": "13.0.0"})
        with patch("online_wearing.runtime.runtime_info", return_value=info):
            with self.assertRaisesRegex(StaleNotebookPackagesError, "Restart the Kaggle") as raised:
                assert_runtime_matches_installed()
        self.assertIn("loaded=13.0.0, installed=12.3.0", str(raised.exception))

    def test_matching_pillow_13_is_accepted(self):
        info = snapshot("13.0.0", {"PIL": "13.0.0", "PIL.Image": "13.0.0", "PIL._imaging": "13.0.0"})
        with patch("online_wearing.runtime.runtime_info", return_value=info):
            self.assertIs(assert_runtime_matches_installed(), info)

    def test_mismatched_core_is_also_detected(self):
        with patch("online_wearing.runtime.runtime_info",
                   return_value=snapshot("13.0.0", {"PIL": "13.0.0", "PIL._imaging": "12.3.0"})):
            with self.assertRaisesRegex(StaleNotebookPackagesError, "PIL._imaging"):
                assert_runtime_matches_installed()

    def test_torch_pypi_version_without_cuda_suffix_is_not_a_false_alarm(self):
        info = {"packages": {"torch": {"installed": "2.7.0", "loaded": {"torch": "2.7.0+cu128"}}}}
        with patch("online_wearing.runtime.runtime_info", return_value=info):
            self.assertIs(assert_runtime_matches_installed(), info)

    def test_no_imports_are_required_to_check_an_unloaded_package(self):
        info = snapshot("13.0.0", {})
        with patch("online_wearing.runtime.runtime_info", return_value=info):
            self.assertIs(assert_runtime_matches_installed(), info)


@unittest.skipIf(Image is None, "Pillow is not installed; no dependency is downloaded by this test")
class ImageDecodingTest(unittest.TestCase):
    def png(self):
        buffer = BytesIO()
        Image.new("RGB", (4, 3), (20, 100, 230)).save(buffer, format="PNG")
        return buffer.getvalue()

    def test_actual_png_bytes_and_hf_mapping_decode(self):
        payload = self.png()
        for value in (payload, bytearray(payload), memoryview(payload), {"bytes": payload, "path": None}):
            with self.subTest(type=type(value).__name__):
                image = data._image(value)
                self.assertEqual((image.size, image.mode), ((4, 3), "RGB"))
                self.assertEqual(image.getpixel((0, 0)), (20, 100, 230))

    def test_path_only_hf_mapping_decodes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "image.png"
            path.write_bytes(self.png())
            self.assertEqual(data._image({"bytes": None, "path": str(path)}).size, (4, 3))

    def test_decoder_explains_stale_pillow_instead_of_reporting_corrupt_data(self):
        info = snapshot("12.3.0", {"PIL": "13.0.0"})
        with patch("PIL.Image.open", side_effect=UnidentifiedImageError("stale plugin")), \
                patch("online_wearing.runtime.runtime_info", return_value=info):
            with self.assertRaises(StaleNotebookPackagesError):
                data._image({"bytes": self.png(), "path": None})

    def test_invalid_source_identifies_row_and_column_without_publishing_manifest(self):
        payload = self.png()
        valid = dict(cloth={"bytes": payload}, person={"bytes": payload}, target={"bytes": payload},
                     **{name: 100. for name in MEASUREMENT_FIELDS})
        invalid = dict(valid, person={"bytes": b"not an image"})
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(data.DatasetImageDecodeError) as raised:
                data.export_rows([valid, invalid], root, limit=2, default_garment_type="vest")
            message = str(raised.exception)
            self.assertIn("sample 000001", message)
            self.assertIn("source_index=1", message)
            self.assertIn("column='person'", message)
            self.assertIn("payload=12 bytes", message)
            self.assertIn("not an image", message)
            self.assertFalse((Path(root) / "manifest.jsonl").exists())
            self.assertEqual(len(data.read_manifest(Path(root) / "manifest.jsonl.partial")), 1)


if __name__ == "__main__":
    unittest.main()
