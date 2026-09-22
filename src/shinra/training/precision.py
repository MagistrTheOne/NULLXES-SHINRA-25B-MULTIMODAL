"""Optional TE GEMMs. Sensitive recurrence/gates remain native FP32 arithmetic."""

from contextlib import contextmanager
from torch import nn


def convert_dense_gemms(model):
    import transformer_engine.pytorch as te

    converted = []
    # Select only backbone Q/K/V/O and SwiGLU matrices with suitable dimensions.
    # Do not quantize decay/beta, norms, text CE, world uncertainty or recurrence.
    paths = []
    for index, block in enumerate(model.layers):
        for name in ("q", "k", "v", "o"):
            paths.append(f"layers.{index}.mixer.{name}")
        if block.kind == "memory":
            paths.append(f"layers.{index}.mixer.output_gate")
        for name in ("gate", "up", "down"):
            paths.append(f"layers.{index}.mlp.{name}")
    for path in paths:
        parent_name, _, name = path.rpartition(".")
        parent = model.get_submodule(parent_name)
        old = getattr(parent, name)
        if not isinstance(old, nn.Linear) or old.in_features % 32 or old.out_features % 32:
            raise ValueError(f"Ineligible low-precision GEMM: {path}")
        new = te.Linear(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            params_dtype=old.weight.dtype,
            device=old.weight.device,
        )
        # Preserve Parameter identity, tying and any optimizer references.
        new.weight = old.weight
        if old.bias is not None:
            new.bias = old.bias
        setattr(parent, name, new)
        parent._shinra_te = True
        converted.append(path)
    model._shinra_te_paths = tuple(converted)
    return model


@contextmanager
def low_precision_context(mode):
    if mode not in ("fp8", "mxfp8"):
        raise ValueError("Expected fp8 or mxfp8")
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format, MXFP8BlockScaling

    recipe = MXFP8BlockScaling() if mode == "mxfp8" else DelayedScaling(fp8_format=Format.HYBRID)
    with te.autocast(enabled=True, recipe=recipe):
        yield
