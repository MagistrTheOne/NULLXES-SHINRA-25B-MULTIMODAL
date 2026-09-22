"""Allocation-free parameter accounting. Does not import PyTorch."""

import json
from .config import ShinraConfig


def parameter_ledger(c: ShinraConfig):
    h, i, depth, d = c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.head_dim
    a, r = c.layer_types.count("global"), c.layer_types.count("memory")
    j, m, g = c.memory_heads, c.memory_heads * c.memory_key_dim, c.memory_gate_rank
    p = {
        "embedding": c.vocab_size * h,
        "attention.q": a * h * h,
        "attention.k": a * h * c.num_key_value_heads * d,
        "attention.v": a * h * c.num_key_value_heads * d,
        "attention.o": a * h * h,
    }
    p.update({f"memory.{x}": r * h * m for x in ("q", "k", "v", "o", "output_gate")})
    p.update(
        {
            "memory.gate_down": r * h * g,
            "memory.decay_up": r * g * m,
            "memory.beta": r * h * j,
            "memory.bias": r * (m + j),
            "memory.A_log": r * j,
            "memory.conv": r * 3 * c.memory_conv_kernel * m,
            "memory.norm": r * m,
        }
    )
    p.update({f"ffn.{x}": depth * h * i for x in ("gate", "up", "down")})
    p.update({"prenorm": 2 * depth * h, "qk_norm": 2 * d * a, "final_norm": h, "lm_head_extra": 0})
    core = sum(p.values())
    z = c.latent_dim

    def t(width, ffn, heads):
        return 4 * width**2 + 3 * width * ffn + 2 * width + 2 * (width // heads)

    def x(width, ffn, heads):
        return t(width, ffn, heads) + width

    patch = 3 * c.patch_size**2
    mm = {
        "visual.blocks": c.visual_layers * t(c.visual_dim, c.visual_ffn, c.visual_heads),
        "visual.patch": patch * c.visual_dim,
        "visual.norm": c.visual_dim,
        "audio.blocks": c.audio_layers * t(c.audio_dim, c.audio_ffn, c.audio_heads),
        "audio.stem": c.audio_stem_kernel * c.audio_dim + 8 * c.audio_dim**2,
        "audio.norm": c.audio_dim,
        "visual.resample_in": c.visual_dim * z,
        "visual.resampler": c.resampler_layers * x(z, c.latent_ffn, c.latent_heads)
        + c.max_visual_latents * z
        + z,
        "audio.resampler": c.resampler_layers * x(z, c.latent_ffn, c.latent_heads)
        + c.audio_query_count * z
        + z,
        "temporal": c.temporal_layers * t(z, c.latent_ffn, c.latent_heads) + z + c.time_features * z,
        "interfaces": 6 * z * h + 6 * z,
        "metadata": (c.modality_types + c.metadata_features) * h,
        "world.slots": c.world_slots * z + c.world_slot_layers * x(z, c.latent_ffn, c.latent_heads) + z,
        "world.future": z * c.latent_ffn + c.latent_ffn * 2 * z,
        "world.action": 4 * c.action_dim * z + c.action_types * z,
        "world.time": c.time_features * z + c.world_horizons * z,
        "world.events": z * c.world_events + 4 * z,
        "visual.decoder": c.visual_decoder_layers * t(z, c.latent_ffn, c.latent_heads)
        + 2 * patch * z
        + z * z
        + c.time_features * z
        + z,
        "audio.decoder": c.audio_decoder_layers
        * t(c.audio_decoder_dim, c.audio_decoder_ffn, c.audio_decoder_heads)
        + 2 * c.audio_output_patch * c.audio_decoder_dim
        + z * 4 * c.audio_decoder_dim
        + c.time_features * c.audio_decoder_dim
        + c.audio_decoder_dim,
    }
    return {
        "core": p,
        "multimodal": mm,
        "core_total": core,
        "multimodal_total": sum(mm.values()),
        "total": core + sum(mm.values()),
    }


def main():
    print(json.dumps(parameter_ledger(ShinraConfig()), indent=2))


if __name__ == "__main__":
    main()
