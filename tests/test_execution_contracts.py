from dataclasses import replace
import pytest
import torch
from shinra.config import ShinraConfig, validate_production_profile
from shinra.kernels import attention, delta_chunk
from shinra.memory import ShinraMemoryMixer
from shinra.model import ShinraForCausalLM
from shinra.multimodal.audio import encoder_frame_hop
from shinra.settings import ShinraRuntimeConfig, ShinraTrainingConfig, execution_segment_size


def test_reference_cap_is_not_context():
    config = ShinraConfig()
    runtime = ShinraRuntimeConfig()
    training = ShinraTrainingConfig()
    assert config.max_context_length == 327680
    assert execution_segment_size(training, runtime, device_type="cpu") == 4096
    assert execution_segment_size(training, runtime, device_type="cuda") == 4096
    assert runtime.reference_backend_max_tokens < config.max_context_length
    assert training.memory_train_segment_size < config.max_context_length


def test_auto_memory_never_selects_kda(config):
    with pytest.raises(ValueError, match="chunk_kda"):
        ShinraRuntimeConfig(memory_backend="fla")
    x = torch.randn(1, 4, config.hidden_size)
    auto = ShinraMemoryMixer(config, runtime=ShinraRuntimeConfig(memory_backend="auto"))
    reference = ShinraMemoryMixer(config, runtime=ShinraRuntimeConfig(memory_backend="reference"))
    reference.load_state_dict(auto.state_dict())
    with torch.no_grad():
        a, sa = auto(x)
        b, sb = reference(x)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(sa.matrix, sb.matrix)
    with pytest.raises(RuntimeError, match="chunk_kda"):
        delta_chunk(x, x, x, x, x, x, "fla")


def test_fused_attention_does_not_inherit_reference_limit():
    q = torch.randn(1, 8, 2, 8)
    with pytest.raises(RuntimeError, match="CUDA"):
        attention(q, q, q, backend="flash2", reference_limit=4)
    with pytest.raises(RuntimeError, match="not the model context"):
        attention(q, q, q, backend="sdpa", reference_limit=4)


def test_split_forwards_match_unsplit_memory(config):
    policy = ShinraTrainingConfig(memory_train_segment_size=4)
    full = ShinraMemoryMixer(config, training=policy)
    split = ShinraMemoryMixer(config, training=policy)
    split.load_state_dict(full.state_dict())
    x = torch.randn(1, 10, config.hidden_size, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    y, state = full(x)
    left, carried = split(other[:, :6])
    right, end = split(other[:, 6:], carried)
    torch.testing.assert_close(torch.cat((left, right), 1), y)
    torch.testing.assert_close(end.matrix, state.matrix)
    torch.testing.assert_close(end.conv, state.conv)
    (y.square().sum() + state.matrix.square().sum() + state.conv.square().sum()).backward()
    (
        left.square().sum() + right.square().sum() + end.matrix.square().sum() + end.conv.square().sum()
    ).backward()
    torch.testing.assert_close(other.grad, x.grad, rtol=2e-4, atol=1e-5)
    for left_p, right_p in zip(full.parameters(), split.parameters()):
        torch.testing.assert_close(left_p.grad, right_p.grad, rtol=2e-4, atol=1e-5)
    reset_y, reset_state = split(other[:, 6:])
    assert not torch.allclose(reset_state.matrix, end.matrix)


def test_model_segments_match_including_cache(config):
    short = ShinraTrainingConfig(memory_train_segment_size=3, mlp_chunk_size=4, loss_chunk_size=3)
    long = replace(short, memory_train_segment_size=64)
    base = ShinraForCausalLM(config, training=long)
    other = ShinraForCausalLM(config, training=short)
    other.load_state_dict(base.state_dict())
    tokens = torch.randint(0, config.vocab_size, (1, 9))
    base_loss = base(tokens, labels=tokens).loss
    other_loss = other(tokens, labels=tokens).loss
    torch.testing.assert_close(base_loss, other_loss, rtol=2e-4, atol=1e-5)
    base_loss.backward()
    other_loss.backward()
    for left, right in zip(base.parameters(), other.parameters()):
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-3, atol=1e-4)
    base.eval()
    other.eval()
    with torch.no_grad():
        a = base(tokens, use_cache=True).cache
        b = other(tokens, use_cache=True).cache
    assert a.length == b.length == 9
    for index in range(config.num_hidden_layers):
        left, right = a.layers[index], b.layers[index]
        if hasattr(left, "matrix"):
            torch.testing.assert_close(left.matrix, right.matrix, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(left.conv, right.conv, rtol=2e-4, atol=2e-5)
        else:
            torch.testing.assert_close(left.key, right.key, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(left.value, right.value, rtol=2e-4, atol=2e-5)
    cleared = a.reset()
    assert cleared.length == 0 and not cleared.layers and not cleared.world


def test_positional_fingerprint_and_production_guard(config):
    changed = replace(config, spatial_rope_theta=12_000.0)
    assert changed.fingerprint() != config.fingerprint()
    from shinra.audit import parameter_ledger

    assert parameter_ledger(config)["total"] == parameter_ledger(changed)["total"]
    with pytest.raises(ValueError, match="Production profile"):
        validate_production_profile(config)
    drifted = replace(ShinraConfig(), layer_types=("global",) * 40)
    with pytest.raises(ValueError, match="topology"):
        validate_production_profile(drifted)
    validate_production_profile(ShinraConfig())


def test_audio_geometry_follows_config(config):
    from shinra.audit import parameter_ledger

    hop = encoder_frame_hop(config)
    assert hop == config.audio_stem_stride * (config.audio_conv_stride**config.audio_stem_stages)
    audio = ShinraForCausalLM(config).multimodal.audio.eval()
    with torch.no_grad():
        latents, times, _ = audio(torch.randn(1, 1024), final=True)
    assert latents.shape[1] == 32 and times.shape[0] == 32
    step = config.audio_frames_per_latent * hop / config.sample_rate
    torch.testing.assert_close(times[1:] - times[:-1], torch.full_like(times[1:], step))
    coarser = replace(config, audio_frames_per_latent=4)
    assert parameter_ledger(config)["total"] == parameter_ledger(coarser)["total"]
    with torch.no_grad():
        fewer, _, _ = ShinraForCausalLM(coarser).multimodal.audio(torch.randn(1, 1024), final=True)
    assert fewer.shape[1] == 16


def test_video_layout_is_in_the_fingerprint(config):
    assert config.video_spatial_tokens == 4 and config.video_fast_tokens == 1
    moved = replace(config, video_fast_tokens=2)
    assert moved.fingerprint() != config.fingerprint()
    mm = ShinraForCausalLM(config).multimodal.eval()
    frames = torch.randn(1, 4, 3, 8, 8)
    with torch.no_grad():
        latents, _, _ = mm.video(frames, torch.arange(4) / 8, final=True)
    # One group: 4 frames * fast tokens, plus one slow resample of spatial tokens.
    assert latents.shape[1] == 4 * config.video_fast_tokens + config.video_spatial_tokens


def test_world_state_horizons_and_counterfactuals(config):
    model = ShinraForCausalLM(config)
    observations = torch.randn(1, 5, config.latent_dim)
    mask = torch.ones(1, config.action_dim)
    kind = torch.tensor([0])
    delta = torch.tensor([0.5])
    action_a = torch.zeros(1, config.action_dim)
    action_b = torch.ones(1, config.action_dim)
    first, _ = model.predict_world(observations, action_a, mask, kind, delta)
    assert first["mean"].shape[2] == len(config.world_horizon_steps)
    assert set(first["aux"]) == set(config.world_aux_channels)
    expected = delta[:, None] * torch.tensor(config.world_horizon_steps, dtype=delta.dtype)
    torch.testing.assert_close(first["horizon_times"], expected)
    first["next_state"].retain_grad()
    second, _ = model.predict_world(
        observations, action_a, mask, kind, delta, previous_state=first["next_state"]
    )
    assert not torch.allclose(first["next_state"], second["next_state"])
    second["next_state"].square().sum().backward()
    assert first["next_state"].grad is not None and first["next_state"].grad.abs().sum() > 0
    assert model.world.action_in.weight.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        prefix = model(torch.tensor([[10, 11]]), use_cache=True).cache
        opened, stored = model.predict_world(observations, action_a, mask, kind, delta, cache=prefix)
        source = stored.world["slots"].clone()
        branch_a, out_a = model.predict_world(observations, action_a, mask, kind, delta, cache=stored)
        branch_b, out_b = model.predict_world(observations, action_b, mask, kind, delta, cache=stored)
        bare = stored.fork()
        bare.world = {}
        without_slots, _ = model.predict_world(observations, action_a, mask, kind, delta, cache=bare)
        torch.testing.assert_close(stored.world["slots"], source)
        assert prefix.length == 2 and "slots" not in prefix.world
        assert out_a is not out_b
        assert not torch.allclose(branch_a["mean"], branch_b["mean"])
        assert not torch.allclose(branch_a["next_state"], branch_b["next_state"])
        assert not torch.allclose(branch_a["next_state"], without_slots["next_state"])
        cleared = stored.reset()
        assert not cleared.world and cleared.length == 0
        restarted, _ = model.predict_world(observations, action_a, mask, kind, delta, cache=cleared)
        fresh, _ = model.predict_world(observations, action_a, mask, kind, delta)
        torch.testing.assert_close(restarted["next_state"], fresh["next_state"])
        assert not torch.allclose(restarted["next_state"], opened["next_state"])
