from dataclasses import replace
import importlib
import pytest
import torch
from shinra.config import ShinraConfig
from shinra.settings import ShinraRuntimeConfig, ShinraTrainingConfig
from shinra.audit import memory_parameter_shapes, memory_ledger, parameter_ledger
from shinra.memory import ShinraMemoryMixer
from shinra.model import ShinraForCausalLM
from shinra.dispatch import select_attention_backend
from shinra.environment import require_pytorch


def test_each_memory_parameter_matches_contract(config):
    mixer = ShinraMemoryMixer(config)
    actual = {name: tuple(p.shape) for name, p in mixer.named_parameters()}
    assert actual == memory_parameter_shapes(config)
    assert sum(p.numel() for p in mixer.parameters()) == memory_ledger(config)["single_mixer_total"]
    full = memory_ledger(ShinraConfig())
    assert full["single_mixer_total"] == 149_971_008
    assert full["all_memory_mixers"] == 4_799_072_256


@pytest.mark.parametrize("segment", [1, 3, 5, 7])
def test_segment_changes_neither_state_nor_gradients(config, segment):
    baseline = ShinraMemoryMixer(config, training=ShinraTrainingConfig(memory_train_segment_size=64))
    segmented = ShinraMemoryMixer(config, training=ShinraTrainingConfig(memory_train_segment_size=segment))
    segmented.load_state_dict(baseline.state_dict())
    x = torch.randn(1, 11, config.hidden_size, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    a, sa = baseline(x)
    b, sb = segmented(other)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(sa.matrix, sb.matrix)
    (a.square().sum() + sa.matrix.square().sum()).backward()
    (b.square().sum() + sb.matrix.square().sum()).backward()
    torch.testing.assert_close(x.grad, other.grad, rtol=2e-4, atol=1e-5)
    for p, q in zip(baseline.parameters(), segmented.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=1e-5)


def test_architecture_hash_excludes_execution_and_loss_policy(config, tmp_path):
    a = ShinraForCausalLM(config)
    b = ShinraForCausalLM(
        config,
        runtime=ShinraRuntimeConfig(attention_backend="sdpa"),
        training=ShinraTrainingConfig(loss_chunk_size=2, z_loss=0.01),
    )
    assert a.config.fingerprint() == b.config.fingerprint()
    assert parameter_ledger(a.config) == parameter_ledger(b.config)
    for obj in (config, a.runtime, a.training_config):
        path = tmp_path / (type(obj).__name__ + ".json")
        obj.save(path)
        assert type(obj).load(path) == obj
    assert ShinraConfig().control_token_range == (130816, 131071)
    assert ShinraConfig().world_latent_dim == 1024
    for key in (
        "attention_backend",
        "memory_backend",
        "memory_train_segment_size",
        "loss_chunk_size",
        "z_loss",
    ):
        assert key not in config.to_dict()


@pytest.mark.parametrize(
    "capability,available,expected",
    [
        ((8, 0), ("flash2", "flash4"), "flash2"),
        ((9, 0), ("flash2", "flash3", "flash4"), "flash3"),
        ((9, 0), ("cudnn", "flash4"), "cudnn"),
        ((10, 3), ("flash2", "flash4", "cudnn"), "flash4"),
    ],
)
def test_hardware_dispatch_policy_without_gpu(capability, available, expected):
    assert select_attention_backend(device="cuda", capability=capability, available=available) == expected


def test_dispatch_has_no_unbounded_math_fallback():
    assert select_attention_backend(device="cpu") == "sdpa"
    with pytest.raises(RuntimeError):
        select_attention_backend(device="cuda", capability=(10, 3), available=())
    with pytest.raises(RuntimeError):
        select_attention_backend(device="cuda", capability=(9, 0), available=("cudnn",), cudnn_eligible=False)


def test_missing_pytorch_diagnostic(monkeypatch):
    original = importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named torch", name="torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="hardware-appropriate PyTorch/CUDA"):
        require_pytorch()


def test_rules_cannot_be_changed_by_labels(config):
    with pytest.raises(ValueError, match="Unsupported architectural contract"):
        replace(config, memory_rule="delta_v2")


def test_audio_query_bank_is_not_a_duration_cap(config):
    model = ShinraForCausalLM(config).eval()
    # 1024 samples / (stem stride 4 * 2 * 2 * 2 frames per latent) = 32 latents.
    with torch.no_grad():
        latents, timestamps, _ = model.multimodal.audio(torch.randn(1, 1024), final=True)
    assert latents.shape[1] == 32 and len(timestamps) == 32
    assert latents.shape[1] > config.audio_query_bank_size


def test_cpu_auto_never_queries_cuda(config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU/reference path must not query CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    model = ShinraForCausalLM(config).eval()
    with torch.no_grad():
        output = model(torch.tensor([[10, 11, 12]]))
    assert output.hidden_states.shape == (1, 3, config.hidden_size)
