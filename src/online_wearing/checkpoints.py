"""Strict tensor-only loading, including common state_dict/module wrappers."""
from collections.abc import Mapping


def load_weights(model, path):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    for wrapper in ("state_dict", "model_state_dict", "model"):
        if isinstance(state, Mapping) and wrapper in state and isinstance(state[wrapper], Mapping):
            state = state[wrapper]
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{path}: expected a nonempty tensor state_dict")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    # Never silently run a partially loaded/random model.
    model.load_state_dict(state, strict=True)
    return model.eval()
