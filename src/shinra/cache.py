"""Explicit inference state. Updates return new cache objects (transactional forward)."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import torch


@dataclass(frozen=True)
class MemoryState:
    matrix: torch.Tensor
    conv: torch.Tensor  # B,3,M,kernel-1


@dataclass(frozen=True)
class GlobalKV:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class ShinraCache:
    fingerprint: str
    version: int = 1
    length: int = 0
    episode: str = "default"
    layers: dict = field(default_factory=dict)
    streams: dict = field(default_factory=dict)
    world: dict = field(default_factory=dict)

    def fork(self):
        # Layer tensors are never mutated in place; branch changes only mappings.
        return ShinraCache(
            self.fingerprint,
            self.version,
            self.length,
            self.episode,
            self.layers.copy(),
            self.streams.copy(),
            self.world.copy(),
        )

    def reset(self, episode="default"):
        return ShinraCache(self.fingerprint, self.version, episode=episode)

    def validate(self, config, additional):
        if self.fingerprint != config.fingerprint() or self.version != config.cache_version:
            raise ValueError("Cache belongs to a different SHINRA contract")
        if additional < 0 or self.length < 0 or self.length + additional > config.max_context_length:
            raise ValueError("Native context budget exceeded")
        if self.length and set(self.layers) != set(range(config.num_hidden_layers)):
            raise ValueError("Incomplete backbone cache")
        if not self.length and self.layers:
            raise ValueError("Empty episode cannot contain backbone states")
        for index, state in self.layers.items():
            if config.layer_types[index] == "memory":
                if not isinstance(state, MemoryState) or state.matrix.dtype != torch.float32:
                    raise ValueError("Memory layer requires an FP32 matrix cache")
                if state.matrix.shape[1:] != (
                    config.memory_heads,
                    config.memory_key_dim,
                    config.memory_value_dim,
                ):
                    raise ValueError("Invalid recurrent cache shape")
                if state.conv.shape[1:] != (
                    3,
                    config.memory_heads * config.memory_key_dim,
                    config.memory_conv_kernel - 1,
                ):
                    raise ValueError("Invalid convolution history shape")
            else:
                if not isinstance(state, GlobalKV) or state.key.shape != state.value.shape:
                    raise ValueError("Invalid global KV cache")
                if state.key.shape[1:] != (self.length, config.num_key_value_heads, config.head_dim):
                    raise ValueError("Invalid global KV cache dimensions")

    def save(self, directory):
        from safetensors.torch import save_file

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        tensors, kinds = {}, {}
        for index, state in self.layers.items():
            prefix = f"layer.{index}."
            kinds[str(index)] = "memory" if isinstance(state, MemoryState) else "global"
            for name in ("matrix", "conv") if isinstance(state, MemoryState) else ("key", "value"):
                tensors[prefix + name] = getattr(state, name).detach().cpu().contiguous()
        for group in ("streams", "world"):
            for key, value in getattr(self, group).items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError("Cache side state must contain named tensors")
                tensors[f"{group}.{key}"] = value.detach().cpu().contiguous()
        save_file(tensors, str(path / "cache.safetensors"))
        meta = {
            "fingerprint": self.fingerprint,
            "version": self.version,
            "length": self.length,
            "episode": self.episode,
            "kinds": kinds,
        }
        (path / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory, config, device="cpu"):
        from safetensors.torch import load_file

        path = Path(directory)
        meta = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        cache = cls(meta["fingerprint"], meta["version"], meta["length"], meta["episode"])
        if cache.fingerprint != config.fingerprint() or cache.version != config.cache_version:
            raise ValueError("Incompatible serialized cache")
        tensors = load_file(str(path / "cache.safetensors"), device=str(device))
        for index, kind in meta["kinds"].items():
            prefix = f"layer.{index}."
            if kind == "memory":
                if tensors[prefix + "matrix"].dtype != torch.float32:
                    raise ValueError("Serialized recurrent state must be FP32")
                state = MemoryState(tensors[prefix + "matrix"], tensors[prefix + "conv"])
            elif kind == "global":
                state = GlobalKV(tensors[prefix + "key"], tensors[prefix + "value"])
                if state.key.shape[1] != cache.length:
                    raise ValueError("Corrupt global KV length")
            else:
                raise ValueError("Unknown cache layer type")
            cache.layers[int(index)] = state
        for name, value in tensors.items():
            group, _, key = name.partition(".")
            if group in ("streams", "world"):
                getattr(cache, group)[key] = value
        if cache.length > config.max_context_length:
            raise ValueError("Serialized cache exceeds context")
        cache.validate(config, 0)
        return cache
