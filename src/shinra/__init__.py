"""NULLXES SHINRA. Importing configuration never imports or allocates a model."""

from .config import ShinraConfig

__all__ = ["ShinraConfig", "ShinraForCausalLM"]


def __getattr__(name):
    if name == "ShinraForCausalLM":
        from .model import ShinraForCausalLM

        return ShinraForCausalLM
    raise AttributeError(name)
