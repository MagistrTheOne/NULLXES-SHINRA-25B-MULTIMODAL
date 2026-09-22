"""Hardware policy only. Pure selection is testable without a CUDA query/workload."""

import importlib.util
from functools import lru_cache


def select_attention_backend(
    *, device, capability=None, dtype="bfloat16", head_dim=128, available=(), cudnn_eligible=True
):
    if device != "cuda":
        return "sdpa"
    if dtype not in {"float16", "bfloat16"} or head_dim % 8 or not 0 < head_dim <= 256:
        raise RuntimeError(
            "No qualified fused attention for this dtype/head shape; use bounded reference explicitly"
        )
    major, _ = capability
    choices = (
        ("flash4", "cudnn", "flash2")
        if major >= 10
        else ("flash3", "cudnn", "flash4", "flash2")
        if major == 9
        else ("flash2", "cudnn")
        if major == 8
        else ()
    )
    for backend in choices:
        if backend in available and (backend != "cudnn" or cudnn_eligible):
            return backend
    raise RuntimeError(
        "No eligible fused attention backend in the compute image; install/qualify it externally"
    )


@lru_cache(maxsize=1)
def installed_attention_backends():
    # Called only when a real CUDA tensor is passed, never during config/audit import.
    found = []
    for name, module in (
        ("flash2", "flash_attn"),
        ("flash3", "flash_attn_interface"),
        ("flash4", "flash_attn.cute.interface"),
    ):
        try:
            if importlib.util.find_spec(module) is not None:
                found.append(name)
        except (ModuleNotFoundError, ValueError):
            continue
    import torch

    if torch.backends.cudnn.is_available():
        found.append("cudnn")
    return tuple(found)
