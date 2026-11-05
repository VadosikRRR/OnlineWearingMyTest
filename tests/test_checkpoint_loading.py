from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_wearing.checkpoints import load_weights


class CheckpointLoadingTest(unittest.TestCase):
    def test_cpu_tensor_only_loading_and_strict_module_prefix_conversion(self):
        tensor = object()
        fake_torch = SimpleNamespace(load=Mock(return_value={"state_dict": {"module.weight": tensor}}))
        model = Mock()
        model.eval.return_value = model
        with patch.dict(sys.modules, torch=fake_torch):
            self.assertIs(load_weights(model, "weights.pt"), model)
        fake_torch.load.assert_called_once_with("weights.pt", map_location="cpu", weights_only=True)
        model.load_state_dict.assert_called_once_with({"weight": tensor}, strict=True)

    def test_empty_checkpoint_rejected_before_model_use(self):
        model = Mock()
        with patch.dict(sys.modules, torch=SimpleNamespace(load=lambda *args, **kwargs: {})):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                load_weights(model, "empty.pt")
        model.eval.assert_not_called()

    def test_architecture_mismatch_propagates(self):
        model = Mock()
        model.load_state_dict.side_effect = RuntimeError("Missing parameter")
        with patch.dict(sys.modules, torch=SimpleNamespace(load=lambda *args, **kwargs: {"wrong.weight": object()})):
            with self.assertRaisesRegex(RuntimeError, "Missing parameter"):
                load_weights(model, "wrong.pt")
        model.eval.assert_not_called()


if __name__ == "__main__":
    unittest.main()
