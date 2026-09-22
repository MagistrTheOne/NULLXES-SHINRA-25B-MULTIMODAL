"""Native SHINRA contract; deliberately independent of Transformers configs."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ShinraConfig:
    architecture_version: int = 2
    family: str = "shinra"
    vocab_size: int = 131072
    hidden_size: int = 7168
    num_hidden_layers: int = 40
    intermediate_size: int = 19456
    num_attention_heads: int = 56
    num_key_value_heads: int = 8
    head_dim: int = 128
    layer_types: tuple[str, ...] = ("memory", "memory", "memory", "memory", "global") * 8
    memory_heads: int = 32
    memory_key_dim: int = 128
    memory_value_dim: int = 128
    memory_gate_rank: int = 256
    memory_conv_kernel: int = 4
    memory_segment_size: int = 8192
    memory_backend: str = "reference"
    attention_backend: str = "sdpa"
    reference_max_tokens: int = 4096
    max_context_length: int = 327680
    rotary_dim: int = 64
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    mlp_chunk_size: int = 2048
    loss_chunk_size: int = 256
    z_loss: float = 1e-5
    gradient_checkpointing: bool = False
    checkpoint_group_size: int = 5
    latent_dim: int = 1024
    visual_dim: int = 1280
    visual_layers: int = 32
    visual_ffn: int = 5120
    visual_heads: int = 20
    patch_size: int = 16
    max_visual_patches: int = 4096
    max_visual_latents: int = 1024
    audio_dim: int = 1024
    audio_layers: int = 24
    audio_ffn: int = 4096
    audio_heads: int = 16
    sample_rate: int = 24000
    audio_stem_kernel: int = 480
    audio_stem_stride: int = 240
    audio_window: int = 256
    audio_query_count: int = 128
    resampler_layers: int = 4
    temporal_layers: int = 4
    latent_heads: int = 16
    latent_ffn: int = 4096
    world_slots: int = 32
    world_slot_layers: int = 2
    action_dim: int = 64
    action_types: int = 64
    world_horizons: int = 8
    world_events: int = 256
    visual_decoder_layers: int = 8
    audio_decoder_dim: int = 768
    audio_decoder_layers: int = 8
    audio_decoder_heads: int = 12
    audio_decoder_ffn: int = 3072
    audio_output_patch: int = 480
    metadata_features: int = 128
    time_features: int = 64
    modality_types: int = 32
    tokenizer_family: str = "sentencepiece_unigram_byte_fallback"
    reserved_control_tokens: int = 256
    cache_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "layer_types", tuple(self.layer_types))
        if self.family != "shinra" or self.architecture_version != 2:
            raise ValueError("Unsupported SHINRA architecture contract")
        if len(self.layer_types) != self.num_hidden_layers or set(self.layer_types) - {"memory", "global"}:
            raise ValueError("Expected one memory/global type per block")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal attention heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        if self.memory_key_dim != self.memory_value_dim:
            raise ValueError("v2 ledger requires equal memory key/value dimensions")
        if self.rotary_dim % 2 or not 0 < self.rotary_dim <= self.head_dim:
            raise ValueError("Invalid rotary dimension")
        if not self.tie_word_embeddings:
            raise ValueError("v2 requires tied embeddings")
        for width, heads in (
            (self.visual_dim, self.visual_heads),
            (self.audio_dim, self.audio_heads),
            (self.latent_dim, self.latent_heads),
            (self.audio_decoder_dim, self.audio_decoder_heads),
        ):
            if width % heads or (width // heads) % 4:
                raise ValueError("Frontend head dimensions must be divisible by four")
        if self.audio_dim != self.latent_dim:
            raise ValueError("Audio resampler contract requires audio_dim == latent_dim")
        if self.attention_backend not in {"sdpa", "flash2", "flash4"}:
            raise ValueError("Unknown attention backend")
        if self.memory_backend not in {"reference", "fla"}:
            raise ValueError("Unknown memory backend")
        for name, value in asdict(self).items():
            if type(value) is int and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.vocab_size < self.reserved_control_tokens:
            raise ValueError("Reserved controls exceed vocabulary")
        if self.memory_conv_kernel < 2 or self.audio_stem_kernel < 2 or self.audio_window < 2:
            raise ValueError("Streaming convolutions/windows require at least two elements")
        if not all(math.isfinite(value) for value in (self.norm_eps, self.rope_theta, self.z_loss)):
            raise ValueError("Nonfinite configuration value")
        if self.norm_eps <= 0 or self.rope_theta <= 1 or self.z_loss < 0:
            raise ValueError("Invalid normalization, positional or loss constant")

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        # Runtime dispatch/checkpoint choices do not alter the serialized weights or cache semantics.
        fields = self.to_dict()
        for key in (
            "memory_backend",
            "attention_backend",
            "reference_max_tokens",
            "gradient_checkpointing",
            "checkpoint_group_size",
            "mlp_chunk_size",
            "loss_chunk_size",
            "memory_segment_size",
        ):
            fields.pop(key)
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
