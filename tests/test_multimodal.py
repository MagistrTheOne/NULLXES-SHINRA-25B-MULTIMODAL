import torch
from shinra.model import ShinraForCausalLM
from shinra.multimodal.decoders import flow_matching_loss, sample_flow
from shinra.losses import future_state_loss


def test_audio_stream_equivalence(config):
    audio = ShinraForCausalLM(config).multimodal.audio.eval()
    wave = torch.randn(1, 173)
    with torch.no_grad():
        expected, times, _ = audio(wave, final=True)
        state = None
        outputs = []
        timestamps = []
        chunks = wave.split(23, dim=1)
        for index, chunk in enumerate(chunks):
            y, t, state = audio(chunk, state, final=index == len(chunks) - 1)
            outputs.append(y)
            timestamps.append(t)
        torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(torch.cat(timestamps), times)


def test_video_stream_equivalence(config):
    mm = ShinraForCausalLM(config).multimodal.eval()
    frames = torch.randn(1, 7, 3, 8, 8)
    times = torch.arange(7) / 8
    with torch.no_grad():
        whole, t, _ = mm.video(frames, times, final=True, spatial_tokens=4, fast_tokens=1)
        first, t1, state = mm.video(frames[:, :3], times[:3], spatial_tokens=4, fast_tokens=1)
        second, t2, _ = mm.video(frames[:, 3:], times[3:], state, final=True, spatial_tokens=4, fast_tokens=1)
        torch.testing.assert_close(torch.cat((first, second), 1), whole)
        torch.testing.assert_close(torch.cat((t1, t2)), t)


def test_modal_and_world_gradients(config):
    model = ShinraForCausalLM(config)
    images = torch.randn(1, 3, 8, 8)
    visual = model.multimodal.image(images, 4)
    audio, _, _ = model.multimodal.audio(torch.randn(1, 128), final=True)
    observations = torch.cat((visual, audio), 1)
    actions = torch.randn(1, config.action_dim)
    args = (
        observations,
        actions,
        torch.ones_like(actions),
        torch.tensor([1]),
        torch.tensor([0.5]),
        torch.tensor([1]),
    )
    world, _ = model.predict_world(*args)
    target = torch.randn_like(world["mean"])
    loss = future_state_loss(world["mean"], world["log_variance"], target)
    loss += flow_matching_loss(model.multimodal.visual_decoder, images, visual)
    waveform = torch.randn(1, audio.shape[1] * 4 * config.audio_output_patch)
    loss += flow_matching_loss(model.multimodal.audio_decoder, waveform, audio)
    loss.backward()
    for parameter in (
        model.multimodal.image.patch.weight,
        model.multimodal.audio.stem[0].weight,
        model.world.action_in.weight,
        model.layers[0].mixer.q.weight,
        model.multimodal.visual_decoder.output.weight,
        model.multimodal.audio_decoder.output.weight,
    ):
        assert (
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
        )


def test_flow_sampler_finite(config):
    decoder = ShinraForCausalLM(config).multimodal.visual_decoder.eval()
    out = sample_flow(decoder, torch.randn(1, 3, 8, 8), torch.randn(1, 4, config.latent_dim), steps=2)
    assert out.shape == (1, 3, 8, 8) and torch.isfinite(out).all()


def test_video_output_objective(config):
    mm = ShinraForCausalLM(config).multimodal
    conditions = torch.randn(1, 2, 4, config.latent_dim, requires_grad=True)
    frames = torch.randn(1, 2, 3, 8, 8)
    times = torch.tensor([0.0, 0.125])
    loss = mm.video_flow_loss(frames, conditions, times)
    loss.backward()
    assert conditions.grad is not None and conditions.grad.abs().sum() > 0
    assert mm.temporal.time.weight.grad.abs().sum() > 0
    assert mm.sample_video(frames, conditions.detach(), times, steps=1).shape == frames.shape


def test_stream_cache_storage_is_bounded(config):
    model = ShinraForCausalLM(config).eval()
    with torch.no_grad():
        cached = model(torch.randint(0, config.vocab_size, (1, 47)), use_cache=True).cache
        for state in cached.layers.values():
            if hasattr(state, "conv"):
                assert state.conv.untyped_storage().nbytes() == state.conv.numel() * state.conv.element_size()
        _, _, audio = model.multimodal.audio(torch.randn(1, 1280), final=False)
        for name, tensor in audio.items():
            if name.endswith((".history", ".k", ".v")) or name == "pending":
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size(), name
        _, _, video = model.multimodal.video(
            torch.randn(1, 8, 3, 8, 8), torch.arange(8) / 8, spatial_tokens=4, fast_tokens=1
        )
        assert video["pending"].untyped_storage().nbytes() == 0
