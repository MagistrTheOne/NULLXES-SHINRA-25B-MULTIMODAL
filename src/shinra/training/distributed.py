"""Explicit distributed setup. No process group is started on import."""

import torch


def wrap_fsdp2(model, mesh, *, cpu_offload=False):
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy

    if mesh.ndim != 1:
        raise ValueError("This FSDP profile requires a one-dimensional data-parallel mesh")
    kwargs = {
        "mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
    }
    if cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
    # Bottom-up wrapping includes modality frontends, not just the language blocks.
    groups = [
        model.layers,
        model.multimodal.image.layers,
        model.multimodal.audio.layers,
        model.multimodal.temporal.layers,
        model.multimodal.visual_decoder.layers,
        model.multimodal.audio_decoder.layers,
        model.multimodal.image.resampler.layers,
        model.multimodal.audio.resampler.layers,
        model.world.slots.layers,
    ]
    for group in groups:
        for block in group:
            fully_shard(block, **kwargs)
    fully_shard(model, **kwargs)
    return model


def initialize_zero_offload(model, *, lr=1e-4, stage=2, gradient_accumulation_steps=1, micro_batch_size=1):
    import deepspeed

    if stage not in (2, 3):
        raise ValueError("Offload profile requires ZeRO stage 2 or 3")
    zero = {
        "stage": stage,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
    }
    if stage == 3:
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
    settings = {
        "bf16": {"enabled": True},
        "zero_optimization": zero,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1},
        },
    }
    return deepspeed.initialize(model=model, model_parameters=model.parameters(), config=settings)


def save_distributed(model, optimizer, directory):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=directory)


def load_distributed(model, optimizer, directory):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_state, optim_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=directory)
    set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
