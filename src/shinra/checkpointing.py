"""Activation checkpoint dispatch, preserving TE quantization recompute state."""

from torch.utils.checkpoint import checkpoint


def activation_checkpoint(function, *args, use_te=False):
    if use_te:
        from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint

        return te_checkpoint(function, *args, use_reentrant=False)
    return checkpoint(function, *args, use_reentrant=False)
