"""Public API. Importing this module does not import ML libraries or select a GPU."""
from .contracts import Weights

__all__ = ["Weights", "OnlineWearingPipeline"]


def __getattr__(name):
    if name == "OnlineWearingPipeline":
        from .pipeline import OnlineWearingPipeline
        return OnlineWearingPipeline
    raise AttributeError(name)
