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
    memory_type: str = "shinra_channel_delta_v1"
    memory_rule: str = "coupled_delta"
    memory_channel_decay: bool = True
    memory_erase_write: str = "shared_scalar_beta"
    memory_output_gate: str = "silu"
    memory_qk_norm: str = "l2"
    memory_output_norm: str = "per_head_rms_affine"
    memory_state_reset: str = "sequence_boundary"
    max_context_length: int = 327680
    rotary_dim: int = 64
    rope_theta: float = 1_000_000.0
    text_position_unit: str = "token_index"
    spatial_rope_theta: float = 10_000.0
    audio_rope_theta: float = 1_000_000.0
    temporal_rope_theta: float = 1_000_000.0
    memory_position: str = "recurrent_only"
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    latent_dim: int = 1024
    visual_dim: int = 1280
    visual_layers: int = 32
    visual_ffn: int = 5120
    visual_heads: int = 20
    visual_window: int = 16
    visual_global_cadence: int = 4
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
    audio_query_bank_size: int = 128
    audio_frames_per_latent: int = 2
    audio_conv_kernel: int = 4
    audio_conv_stride: int = 2
    audio_stem_stages: int = 2
    audio_decoder_expand: int = 4
    video_spatial_tokens: int = 64
    video_fast_tokens: int = 8
    video_frame_group: int = 4
    video_position: str = "frame_xy_plus_availability_time"
    resampler_layers: int = 4
    temporal_layers: int = 4
    latent_heads: int = 16
    latent_ffn: int = 4096
    world_slots: int = 32
    world_slot_layers: int = 2
    action_dim: int = 64
    action_types: int = 64
    world_horizons: int = 8
    world_horizon_steps: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    world_aux_channels: tuple[str, ...] = ("reward", "terminal", "value", "cost")
    world_events: int = 256
    world_continuous_time: bool = True
    world_counterfactual: bool = True
    world_state_reset: str = "episode_boundary"
    vision_generation: bool = True
    audio_generation: bool = True
    video_generation: str = "latent_conditioned_shared_visual_flow"
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
    cache_version: int = 2
    memory_contract_version: int = 2
    world_contract_version: int = 2
    experimental_hypotheses: tuple[str, ...] = ("rotary_dim", "memory_gate_rank")

    def __post_init__(self):
        object.__setattr__(self, "layer_types", tuple(self.layer_types))
        object.__setattr__(self, "world_horizon_steps", tuple(self.world_horizon_steps))
        object.__setattr__(self, "world_aux_channels", tuple(self.world_aux_channels))
        object.__setattr__(self, "experimental_hypotheses", tuple(self.experimental_hypotheses))
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
        contract = {
            "memory_type": "shinra_channel_delta_v1",
            "memory_rule": "coupled_delta",
            "memory_erase_write": "shared_scalar_beta",
            "memory_output_gate": "silu",
            "memory_qk_norm": "l2",
            "memory_output_norm": "per_head_rms_affine",
            "memory_state_reset": "sequence_boundary",
            "memory_position": "recurrent_only",
            "text_position_unit": "token_index",
            "video_position": "frame_xy_plus_availability_time",
            "world_state_reset": "episode_boundary",
            "video_generation": "latent_conditioned_shared_visual_flow",
        }
        if any(getattr(self, key) != value for key, value in contract.items()):
            raise ValueError(
                "Unsupported architectural contract; changing a name does not implement a new rule"
            )
        if not all(
            (
                self.memory_channel_decay,
                self.world_continuous_time,
                self.world_counterfactual,
                self.vision_generation,
                self.audio_generation,
            )
        ):
            raise ValueError("Unsupported modality/memory contract")
        if self.experimental_hypotheses != ("rotary_dim", "memory_gate_rank"):
            raise ValueError("Experimental hypothesis labels are part of the v2 contract")
        if (self.cache_version, self.memory_contract_version, self.world_contract_version) != (2, 2, 2):
            raise ValueError("Execution contract versions are part of architecture v2")
        if len(self.world_horizon_steps) != self.world_horizons or len(set(self.world_horizon_steps)) != len(
            self.world_horizon_steps
        ):
            raise ValueError("Each world horizon needs one unique temporal offset")
        if tuple(sorted(self.world_horizon_steps)) != self.world_horizon_steps:
            raise ValueError("World horizon offsets must be strictly increasing")
        if any(step <= 0 for step in self.world_horizon_steps):
            raise ValueError("World horizon offsets must be positive")
        if not self.world_aux_channels or len(set(self.world_aux_channels)) != len(self.world_aux_channels):
            raise ValueError("World auxiliary channels must be unique and named")
        if (
            self.video_spatial_tokens > self.max_visual_latents
            or self.video_fast_tokens > self.max_visual_latents
        ):
            raise ValueError("Video token layout exceeds the visual query bank")
        for name, value in asdict(self).items():
            if type(value) is int and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.vocab_size < self.reserved_control_tokens:
            raise ValueError("Reserved controls exceed vocabulary")
        if (
            self.memory_conv_kernel < 2
            or self.audio_stem_kernel < 2
            or self.audio_conv_kernel < 2
            or self.audio_window < 2
        ):
            raise ValueError("Streaming convolutions/windows require at least two elements")
        thetas = (self.rope_theta, self.spatial_rope_theta, self.audio_rope_theta, self.temporal_rope_theta)
        if not all(math.isfinite(value) for value in (self.norm_eps, *thetas)):
            raise ValueError("Nonfinite configuration value")
        if self.norm_eps <= 0 or any(value <= 1 for value in thetas):
            raise ValueError("Invalid normalization, positional or loss constant")

    def to_dict(self):
        return asdict(self)

    @property
    def control_token_range(self):
        """Inclusive range inside vocab_size, never appended to it."""
        return self.vocab_size - self.reserved_control_tokens, self.vocab_size - 1

    @property
    def world_latent_dim(self):
        return self.latent_dim

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def validate_production_profile(config: ShinraConfig):
    """Accepted 25B baseline. Tiny fixtures are not required to pass."""
    pattern = ("memory", "memory", "memory", "memory", "global") * 8
    expected = {
        "hidden_size": 7168,
        "num_hidden_layers": 40,
        "intermediate_size": 19456,
        "num_attention_heads": 56,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_context_length": 327680,
        "vocab_size": 131072,
        "rotary_dim": 64,
        "rope_theta": 1_000_000.0,
        "spatial_rope_theta": 10_000.0,
        "audio_rope_theta": 1_000_000.0,
        "temporal_rope_theta": 1_000_000.0,
        "memory_gate_rank": 256,
        "memory_heads": 32,
        "memory_key_dim": 128,
        "memory_value_dim": 128,
        "memory_conv_kernel": 4,
        "visual_dim": 1280,
        "visual_layers": 32,
        "visual_window": 16,
        "visual_global_cadence": 4,
        "audio_dim": 1024,
        "audio_layers": 24,
        "audio_stem_kernel": 480,
        "audio_stem_stride": 240,
        "audio_conv_kernel": 4,
        "audio_conv_stride": 2,
        "audio_stem_stages": 2,
        "audio_frames_per_latent": 2,
        "audio_decoder_expand": 4,
        "video_spatial_tokens": 64,
        "video_fast_tokens": 8,
        "video_frame_group": 4,
        "latent_dim": 1024,
        "world_slots": 32,
        "world_horizons": 8,
        "world_horizon_steps": (1, 2, 4, 8, 16, 32, 64, 128),
        "world_aux_channels": ("reward", "terminal", "value", "cost"),
    }
    problems = []
    memory = config.layer_types.count("memory")
    global_layers = config.layer_types.count("global")
    if config.layer_types != pattern or memory != 32 or global_layers != 8:
        problems.append("topology is not 32 memory + 8 global as MMMMG x 8")
    for key, value in expected.items():
        actual = getattr(config, key)
        if actual != value:
            problems.append(f"{key}={actual!r}, expected {value!r}")
    if problems:
        raise ValueError("Production profile mismatch: " + "; ".join(problems))


def validate_production_stack(config, runtime, training):
    validate_production_profile(config)
    if runtime.attention_backend != "auto" or runtime.memory_backend != "auto":
        raise ValueError("Published runtime dispatch is auto for attention and memory")
    if runtime.reference_backend_max_tokens >= config.max_context_length:
        raise ValueError(
            "reference_backend_max_tokens is a reference-backend cap and must stay below max_context_length"
        )
    if training.memory_train_segment_size >= config.max_context_length:
        raise ValueError(
            "memory_train_segment_size partitions execution and must stay below max_context_length"
        )
