import json
import pytest
import torch
from shinra.model import ShinraForCausalLM
from shinra.serialization import save_checkpoint, load_checkpoint
from shinra.multimodal.stream import TextSegment, LatentSegment, assemble_stream
from shinra.training.curriculum import ReplaySampler, CapabilityCycle, regression_gate
from shinra.training.offload import CPUAdamW
from shinra.evaluation import effective_context
from shinra.settings import ShinraRuntimeConfig


def test_checkpoint_roundtrip_and_corruption(config, tmp_path):
    model = ShinraForCausalLM(config)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    save_checkpoint(model, tmp_path / "checkpoint", training_state={"updates": 7}, shard_bytes=18000)
    with torch.no_grad():
        model.embed_tokens.weight.add_(1)
    state = load_checkpoint(model, tmp_path / "checkpoint")
    assert state == {"updates": 7}
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, original[name])
    with pytest.raises(FileExistsError):
        save_checkpoint(model, tmp_path / "checkpoint")
    shard = next((tmp_path / "checkpoint").glob("*.safetensors"))
    with shard.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(model, tmp_path / "checkpoint")


def test_stream_mask_and_backprop(config):
    model = ShinraForCausalLM(config)
    latents = model.multimodal.image(torch.randn(1, 3, 8, 8), 4)
    segments = [
        TextSegment(torch.tensor([[10, 11, 12]])),
        LatentSegment(latents, "visual", 1, 56, 57, torch.zeros(1, 4, 3)),
        TextSegment(torch.tensor([[13, 14]])),
    ]
    stream = assemble_stream(model, segments)
    assert stream.inputs_embeds.shape == (1, 11, config.hidden_size)
    assert (stream.labels[:, 4:8] == -100).all()
    model(inputs_embeds=stream.inputs_embeds, labels=stream.labels).loss.backward()
    assert model.multimodal.interfaces["visual"].input.weight.grad.abs().sum() > 0


def test_causal_prefix_independence(config):
    model = ShinraForCausalLM(config).eval()
    first = torch.randint(0, config.vocab_size, (1, 12))
    second = first.clone()
    second[:, 7:] = (second[:, 7:] + 1) % config.vocab_size
    with torch.no_grad():
        a, b = model(first).hidden_states, model(second).hidden_states
    torch.testing.assert_close(a[:, :7], b[:, :7], rtol=1e-5, atol=1e-6)


def test_context_and_reference_guards(config):
    model = ShinraForCausalLM(config)
    with pytest.raises(ValueError, match="context"):
        model(torch.zeros(1, 129, dtype=torch.long))
    guarded = ShinraForCausalLM(config, runtime=ShinraRuntimeConfig(reference_backend_max_tokens=4))
    with pytest.raises(RuntimeError, match="Reference recurrence"):
        guarded(torch.zeros(1, 5, dtype=torch.long))


def test_cpu_gradient_offload_accumulation():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    optimizer = CPUAdamW([parameter], gradient_offload=True)
    parameter.square().sum().backward()
    parameter.sum().backward()
    assert parameter.grad is None
    torch.testing.assert_close(optimizer.masters[0].grad, torch.tensor([5.0, 7.0]))
    optimizer.zero_grad()
    assert optimizer.masters[0].grad is None
    optimizer.close()


def test_replay_resume_and_regression():
    a = ReplaySampler(([1, 2], [3, 4]), [5, 6], seed=7)
    for _ in range(5):
        a.draw()
    state = json.loads(json.dumps(a.state_dict()))
    expected = [a.draw() for _ in range(20)]
    b = ReplaySampler(([1, 2], [3, 4]), [5, 6], seed=19)
    b.load_state_dict(state)
    assert [b.draw() for _ in range(20)] == expected
    assert not regression_gate({"reasoning": 0.9}, {"reasoning": 0.7}, {"reasoning": 0.05})["accepted"]
    assert effective_context({8192: {"retrieve": 1.0}, 327680: {"retrieve": 0.4}}) == 8192
    with pytest.raises(ValueError):
        CapabilityCycle("invalid", ("same", "same"))
