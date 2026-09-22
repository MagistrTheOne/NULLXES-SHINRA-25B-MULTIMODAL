import json
from pathlib import Path
import pytest
import torch
from shinra import ShinraConfig
from shinra.audit import parameter_ledger
from shinra.model import ShinraForCausalLM
from shinra.training.offload import CPUAdamW
from shinra.settings import ShinraRuntimeConfig, ShinraTrainingConfig


def test_published_config_and_ledger():
    root = Path(__file__).parents[1]
    config = ShinraConfig.load(root / "configs/shinra25b.json")
    assert config.max_context_length == 327680
    runtime = ShinraRuntimeConfig.load(root / "configs/runtime.json")
    training = ShinraTrainingConfig.load(root / "configs/training.json")
    assert runtime.attention_backend == "auto" and runtime.memory_backend == "auto"
    assert training.gradient_checkpointing and training.memory_train_segment_size == 8192
    assert parameter_ledger(config) == json.loads((root / "specifications/parameter_ledger.json").read_text())


def test_hf_adapter_matches_native(config):
    pytest.importorskip("transformers")
    from shinra.compatibility.huggingface import ShinraHFConfig, ShinraHFForCausalLM

    adapter = ShinraHFForCausalLM(ShinraHFConfig(shinra=config.to_dict())).eval()
    ids = torch.tensor([[10, 11, 12, 13]])
    with torch.no_grad():
        wrapped = adapter(input_ids=ids, labels=ids)
        native = adapter.shinra(ids, labels=ids, logits_to_keep=1)
        torch.testing.assert_close(wrapped.loss, native.loss)
        torch.testing.assert_close(wrapped.logits, native.logits)
    with pytest.raises(ValueError, match="Padded"):
        adapter(input_ids=ids, attention_mask=torch.tensor([[1, 1, 1, 0]]))


def test_generation_matches_manual_greedy(config):
    model = ShinraForCausalLM(config).eval()
    ids = torch.tensor([[10, 11, 12]])
    with torch.no_grad():
        expected = ids
        for _ in range(4):
            token = model(expected, logits_to_keep=1).logits[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, token), 1)
        torch.testing.assert_close(model.generate(ids, max_new_tokens=4), expected)


def test_offloaded_model_gradients_match(config):
    ordinary = ShinraForCausalLM(config)
    offloaded = ShinraForCausalLM(config)
    offloaded.load_state_dict(ordinary.state_dict())
    optimizer = CPUAdamW(offloaded.parameters(), gradient_offload=True)
    ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    ordinary(ids, labels=ids).loss.backward()
    offloaded(ids, labels=ids).loss.backward()
    for expected, actual in zip(ordinary.parameters(), optimizer.masters):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6)
    optimizer.close()


def test_world_branches_preserve_prefix(config):
    model = ShinraForCausalLM(config).eval()
    with torch.no_grad():
        prefix = model(torch.tensor([[10, 11]]), use_cache=True).cache
        observations = torch.randn(1, 4, config.latent_dim)
        actions = torch.zeros(1, config.action_dim)
        common = (torch.ones_like(actions), torch.tensor([0]), torch.tensor([1.0]), torch.tensor([0]))
        first, a = model.predict_world(observations, actions, *common, cache=prefix)
        second, b = model.predict_world(observations, actions + 1, *common, cache=prefix)
    assert prefix.length == 2
    assert a.length == b.length == 2 + config.world_slots
    assert not torch.allclose(first["mean"], second["mean"])
    assert "slots" in a.world and "slots" not in prefix.world
