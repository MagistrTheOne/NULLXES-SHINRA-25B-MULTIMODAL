"""NULLXES SHINRA. Importing configuration never imports or allocates a model."""

from .config import ShinraConfig
from .settings import ShinraRuntimeConfig, ShinraTrainingConfig

__all__ = ["ShinraConfig", "ShinraForCausalLM", "ShinraRuntimeConfig", "ShinraTrainingConfig"]


def __getattr__(name):
    if name == "ShinraForCausalLM":
        from .environment import require_pytorch

        require_pytorch()
        from .model import ShinraForCausalLM

        return ShinraForCausalLM
    raise AttributeError(name)
