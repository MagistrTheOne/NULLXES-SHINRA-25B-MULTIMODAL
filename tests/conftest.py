from dataclasses import replace
import pytest
import torch
from shinra.config import ShinraConfig


@pytest.fixture
def config():
    # CPU numerical fixture, not a second architecture or a released model config.
    return replace(
        ShinraConfig(),
        vocab_size=64,
        reserved_control_tokens=8,
        hidden_size=32,
        num_hidden_layers=5,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=("memory", "memory", "memory", "memory", "global"),
        memory_heads=2,
        memory_key_dim=8,
        memory_value_dim=8,
        memory_gate_rank=8,
        max_context_length=128,
        rotary_dim=8,
        latent_dim=16,
        visual_dim=16,
        visual_layers=2,
        visual_ffn=32,
        visual_heads=2,
        visual_window=16,
        visual_global_cadence=4,
        patch_size=2,
        max_visual_patches=64,
        max_visual_latents=16,
        audio_dim=16,
        audio_layers=2,
        audio_ffn=32,
        audio_heads=2,
        sample_rate=400,
        audio_stem_kernel=8,
        audio_stem_stride=4,
        audio_window=8,
        audio_query_bank_size=8,
        video_spatial_tokens=4,
        video_fast_tokens=1,
        video_frame_group=4,
        resampler_layers=2,
        temporal_layers=2,
        latent_heads=2,
        latent_ffn=32,
        world_slots=4,
        world_slot_layers=2,
        action_dim=3,
        action_types=4,
        world_horizons=2,
        world_horizon_steps=(1, 4),
        world_events=5,
        visual_decoder_layers=2,
        audio_decoder_dim=16,
        audio_decoder_layers=2,
        audio_decoder_heads=2,
        audio_decoder_ffn=32,
        audio_output_patch=8,
        metadata_features=8,
        time_features=8,
        modality_types=4,
    )


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(319)
