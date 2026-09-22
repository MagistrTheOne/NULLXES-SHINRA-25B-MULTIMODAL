from contextlib import nullcontext
from dataclasses import dataclass
import torch
from .offload import CPUAdamW


@dataclass(frozen=True)
class RuntimeConfig:
    precision: str = "bf16"
    activation_offload: bool = False
    grad_clip: float = 1.0
    compile_model: bool = False
    data_parallel: bool = False

    def __post_init__(self):
        if self.precision not in ("bf16", "fp32", "fp8", "mxfp8") or self.grad_clip <= 0:
            raise ValueError("Invalid runtime precision or gradient clipping")


def prepare_runtime(model, config):
    if config.precision in ("fp8", "mxfp8"):
        from .precision import convert_dense_gemms

        model = convert_dense_gemms(model)
    return torch.compile(model, dynamic=False) if config.compile_model else model


def optimizer_update(model, optimizer, microbatches, config=RuntimeConfig(), objective=None):
    """One explicitly invoked update. Never called by import/CLI/audit.

    Text batches are normalized over valid targets across all microbatches.
    Custom objectives return (loss_sum, valid_count), with counts supplied in batch.
    """
    batches = list(microbatches)
    if not batches:
        raise ValueError("Empty optimizer update")
    counts = [
        int(batch["valid_count"]) if objective is not None else int((batch["labels"][:, 1:] != -100).sum())
        for batch in batches
    ]
    device = next(model.parameters()).device
    total = sum(counts)
    world_size = 1
    if config.data_parallel:
        import torch.distributed as dist

        if not dist.is_initialized() or isinstance(optimizer, CPUAdamW):
            raise ValueError("Data parallel updates require an initialized group and its sharded optimizer")
        world_size = dist.get_world_size()
        # DDP/FSDP average gradients; multiply local numerator by world size.
        count_tensor = torch.tensor(total, device=device, dtype=torch.int64)
        dist.all_reduce(count_tensor)
        total = int(count_tensor)
    if total == 0:
        raise ValueError("Update has no supervised targets")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum = 0.0
    for batch, count in zip(batches, counts):
        autocast = (
            torch.autocast(device.type, dtype=torch.bfloat16) if config.precision != "fp32" else nullcontext()
        )
        offload = (
            torch.autograd.graph.save_on_cpu(pin_memory=device.type == "cuda")
            if config.activation_offload
            else nullcontext()
        )
        if config.precision in ("fp8", "mxfp8"):
            from .precision import low_precision_context

            precision_context = low_precision_context(config.precision)
        else:
            precision_context = nullcontext()
        with autocast, offload, precision_context:
            if objective is None:
                args = {
                    key: value
                    for key, value in batch.items()
                    if key in ("input_ids", "inputs_embeds", "labels")
                }
                output = model(**args)
                loss = output.loss * count * world_size / total
            else:
                loss_sum, actual_count = objective(model, batch)
                if int(actual_count) != count:
                    raise ValueError("Objective count differs from update normalization")
                loss = loss_sum * world_size / total
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Nonfinite loss; optimizer was not advanced")
        loss.backward()
        accum += float(loss.detach())
    norm = (
        optimizer.clip_grad_norm(config.grad_clip)
        if isinstance(optimizer, CPUAdamW)
        else torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
    )
    optimizer.step()
    if config.data_parallel:
        metric = torch.tensor(accum, device=device)
        dist.all_reduce(metric)
        accum = float(metric) / world_size
    if hasattr(norm, "full_tensor"):
        norm = norm.full_tensor()
    return {"loss": accum, "valid_targets": total, "grad_norm": float(norm)}
