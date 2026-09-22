"""Diagnostics only: never install or modify the compute environment."""

import importlib


def require_pytorch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise RuntimeError(
            "Install the hardware-appropriate PyTorch/CUDA environment before SHINRA."
        ) from error
