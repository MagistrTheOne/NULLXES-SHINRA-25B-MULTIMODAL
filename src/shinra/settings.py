"""Execution/training settings are not part of the model architecture or its hash."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


class SettingsFile:
    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ShinraRuntimeConfig(SettingsFile):
    attention_backend: str = "auto"
    memory_backend: str = "auto"
    reference_backend_max_tokens: int = 4096
    compile_model: bool = False
    cache_dtype: str = "activation"

    def __post_init__(self):
        if self.attention_backend not in {"auto", "sdpa", "flash2", "flash3", "flash4", "cudnn"}:
            raise ValueError("Unknown attention backend")
        if self.memory_backend not in {"auto", "reference", "fla"}:
            raise ValueError("Unknown memory backend")
        if self.reference_backend_max_tokens < 1:
            raise ValueError("Reference correctness-backend limit must be positive")
        if self.cache_dtype != "activation":
            raise ValueError("Only activation-dtype KV is implemented; memory state remains FP32")


@dataclass(frozen=True)
class ShinraTrainingConfig(SettingsFile):
    gradient_checkpointing: bool = False
    checkpoint_group_size: int = 5
    mlp_chunk_size: int = 2048
    loss_chunk_size: int = 256
    memory_train_segment_size: int = 8192
    z_loss: float = 1e-5
    precision: str = "bf16"
    activation_offload: bool = False
    grad_clip: float = 1.0
    data_parallel: bool = False

    def __post_init__(self):
        for name in (
            "checkpoint_group_size",
            "mlp_chunk_size",
            "loss_chunk_size",
            "memory_train_segment_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.precision not in {"bf16", "fp32", "fp8", "mxfp8"}:
            raise ValueError("Invalid training precision")
        if (
            not math.isfinite(self.z_loss)
            or self.z_loss < 0
            or not math.isfinite(self.grad_clip)
            or self.grad_clip <= 0
        ):
            raise ValueError("Invalid training loss/clipping policy")
