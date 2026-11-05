from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_wearing.contracts import (Weights, normalize_garment_type, LANDMARK_MAPS,
                                      LANDMARK_COUNTS, LONG_TO_KGI, VEST_TO_KGI,
                                      BODY25_TO_KGI, COCO18_TO_KGI, LIP_TO_KGI,
                                      CONTOUR_EDGES, TPS_GROUPS)
from online_wearing.vendor.sci.scheduler import get_schedule_jump


class ContractsTest(unittest.TestCase):
    def test_landmark_mapping_cardinality_and_bounds(self):
        for name, indices in LANDMARK_MAPS.items():
            self.assertEqual(len(indices), 32)
            self.assertGreaterEqual(min(indices), 0)
            self.assertLess(max(indices), LANDMARK_COUNTS[name])
        self.assertEqual(LONG_TO_KGI, tuple(range(1, 33)))
        self.assertEqual(VEST_TO_KGI[5:10], (6,) * 5)
        self.assertEqual(VEST_TO_KGI[10:16], (7,) * 6)

    def test_pose_hips_have_explicit_schema_conversion(self):
        self.assertEqual(BODY25_TO_KGI[:8], COCO18_TO_KGI[:8])
        self.assertEqual(BODY25_TO_KGI[-2:], (9, 12))
        self.assertEqual(COCO18_TO_KGI[-2:], (8, 11))

    def test_lip_arms_socks_and_clothing(self):
        self.assertEqual(tuple(LIP_TO_KGI[i] for i in [5, 6, 7]), (3, 3, 3))
        self.assertEqual(tuple(LIP_TO_KGI[i] for i in [14, 15]), (5, 6))
        self.assertEqual(LIP_TO_KGI[8], 11)
        self.assertEqual(set(LIP_TO_KGI), set(range(13)))

    def test_contour_is_one_closed_32_node_cycle(self):
        self.assertEqual(len(CONTOUR_EDGES), 32)
        degrees = [0] * 32
        for left, right in CONTOUR_EDGES:
            degrees[left] += 1
            degrees[right] += 1
        self.assertEqual(degrees, [2] * 32)
        self.assertEqual(len(TPS_GROUPS), 5)

    def test_categories_and_unsupported_lower_body(self):
        self.assertEqual(normalize_garment_type(" ФУТБОЛКА "), "short sleeve top")
        for category in ("dress", "trousers", "skirt", "anything"):
            with self.assertRaisesRegex(ValueError, "retraining"):
                normalize_garment_type(category)

    def test_all_weights_checked_together(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "one.pt"
            existing.write_bytes(b"test")
            weights = Weights(existing, existing, existing, existing, existing, existing)
            weights.validate()
            self.assertEqual(len(weights.as_dict()), 6)
            missing = Weights(*[Path(directory) / f"{i}.pt" for i in range(6)])
            with self.assertRaises(FileNotFoundError) as raised:
                missing.validate()
            for name in ("kg", "pg", "sci", "garment_hrnet", "openpose_body", "human_parser_lip"):
                self.assertIn(name, str(raised.exception))

    def test_repaint_schedule_bounds_for_preview_and_original(self):
        for steps, repeats in ((2, 1), (10, 1), (50, 10)):
            times = get_schedule_jump(steps, 1, min(10, steps - 1), repeats)
            self.assertEqual(times[0], steps - 1)
            self.assertEqual(times[-1], -1)
            self.assertLess(max(times), steps)
            self.assertTrue(all(abs(a - b) == 1 for a, b in zip(times[:-1], times[1:])))


if __name__ == "__main__":
    unittest.main()
