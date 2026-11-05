"""Kaggle-only CPU contracts/TPS/forward smoke tests; no weights or network."""
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from types import SimpleNamespace
from online_wearing.contracts import LIP_TO_KGI
from online_wearing.preprocessing import lip_to_kgi, landmarks_to_kgi, HumanParser, HumanPose
from online_wearing.data import _image
from online_wearing.runtime import assert_runtime_matches_installed
from online_wearing.pipeline import adjacency, one_hot
from online_wearing.warping import inverse_tps
from online_wearing.contracts import CONTOUR_EDGES, SYMMETRY_EDGES, POSE_EDGES
from online_wearing.vendor.kg import GCN_2
from online_wearing.vendor.pg import ParseGenerator
from online_wearing.vendor.sci.unet import UNetModel


def main():
    assert_runtime_matches_installed()
    buffer = BytesIO()
    Image.new("RGB", (4, 3), (20, 100, 230)).save(buffer, format="PNG")
    image = _image({"bytes": buffer.getvalue(), "path": None})
    assert image.size == (4, 3) and image.getpixel((0, 0)) == (20, 100, 230)
    labels = np.arange(20, dtype=np.uint8)[None]
    np.testing.assert_array_equal(lip_to_kgi(labels), np.array(LIP_TO_KGI)[None])
    points = np.array([[0, 0], [.999, 0], [0, .999], [.999, .999]], dtype=np.float32)
    map_x, map_y = inverse_tps(points, points, 32, 24)
    expected_y, expected_x = np.mgrid[:32, :24]
    np.testing.assert_allclose(map_x, expected_x, atol=1e-4)
    np.testing.assert_allclose(map_y, expected_y, atol=1e-4)
    try:
        inverse_tps(np.zeros((4, 2)), points, 32, 24)
    except ValueError:
        pass
    else:
        raise AssertionError("Degenerate TPS must fail")
    mapped = landmarks_to_kgi(np.tile([384, 512], (33, 1)), "long sleeve top")
    np.testing.assert_allclose(mapped, .5)
    assert one_hot(np.array([[3, 5]], np.uint8), "cpu").shape == (1, 13, 1, 2)

    # Validate the ONNX output contract/affine recovery without loading any weights.
    class FakeSession:
        channels = 20

        def get_inputs(self):
            return [SimpleNamespace(name="input.1")]

        def run(self, outputs, inputs):
            assert inputs["input.1"].shape == (1, 3, 512, 512)
            logits = np.zeros((1, self.channels, 16, 16), np.float32)
            logits[:, 5] = 1
            return [logits]

    parser = HumanParser.__new__(HumanParser)
    parser.session = FakeSession()
    lip, kgi = parser.predict(np.full((100, 60, 3), 128, np.uint8))
    assert lip.shape == kgi.shape == (100, 60)
    assert np.all(lip == 5) and np.all(kgi == 3)
    parser.session.channels = 18
    try:
        parser.predict(np.full((100, 60, 3), 128, np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError("ATR-18 must not be accepted as LIP-20")

    class FakeBody:
        def to(self, device):
            return self

        def __call__(self, image):
            return None, None

        def format_body_result(self, candidates, subsets):
            return [SimpleNamespace(keypoints=[SimpleNamespace(x=i * 10, y=100) for i in range(18)])]

    pose = HumanPose.__new__(HumanPose)
    pose.body = FakeBody()
    points, missing = pose.predict(np.zeros((1024, 768, 3), np.uint8), "cpu")
    np.testing.assert_allclose(points[-2:, 0], np.array([80, 110]) / 768, atol=1e-6)
    assert missing == []
    with torch.no_grad():
        kg = GCN_2(adjacency(32, CONTOUR_EDGES + SYMMETRY_EDGES), adjacency(10, POSE_EDGES), 160).eval()
        assert kg(torch.zeros(1, 32, 2), torch.zeros(1, 10, 2)).shape == (1, 32, 2)
        # Reduced image only for an architecture smoke check, not pretrained inference.
        pg = ParseGenerator(19, 13, ngf=8).eval()
        result = pg(torch.zeros(1, 19, 64, 64))
        assert result.shape == (1, 13, 64, 64)
        sci = UNetModel(image_size=64, in_channels=3, model_channels=32, out_channels=6,
                        num_res_blocks=1, attention_resolutions=(4,), channel_mult=(1, 2),
                        num_classes=13, num_heads=1, num_head_channels=32,
                        use_checkpoint=False, use_scale_shift_norm=True, resblock_updown=True).eval()
        result = sci(torch.zeros(1, 3, 64, 48), torch.tensor([1]), torch.zeros(1, 13, 64, 48))
        assert result.shape == (1, 6, 64, 48)
    print("PASS: Pillow PNG decode, CPU LIP/ONNX/pose adapters, landmarks, TPS, KG/PG/SCI forward contracts")


if __name__ == "__main__":
    main()
