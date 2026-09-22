import tomllib
from pathlib import Path
from dataclasses import replace
import pytest
from shinra.config import ShinraConfig
from shinra.audit import parameter_ledger
from shinra.model import ShinraForCausalLM


def test_default_exact_ledger_without_allocation():
    ledger = parameter_ledger(ShinraConfig())
    assert ledger["core_total"] == 23_413_976_064
    assert ledger["multimodal_total"] == 1_763_842_816
    assert ledger["total"] == 25_177_818_880


def test_actual_parameter_contract(config):
    model = ShinraForCausalLM(config)
    ledger = parameter_ledger(config)
    assert sum(p.numel() for p in model.parameters()) == ledger["total"]
    assert (
        sum(p.numel() for p in model.multimodal.parameters())
        + sum(p.numel() for p in model.world.parameters())
        == ledger["multimodal_total"]
    )
    assert model.lm_head_weight is model.embed_tokens.weight


def test_large_allocation_guard():
    with pytest.raises(RuntimeError, match="Large model allocation"):
        ShinraForCausalLM()


def test_config_roundtrip(config, tmp_path):
    path = tmp_path / "config.json"
    config.save(path)
    assert ShinraConfig.load(path) == config
    assert "attention_backend" not in config.to_dict()
    with pytest.raises(ValueError):
        replace(config, num_key_value_heads=3)


def test_no_runtime_dependency_resolution():
    doc = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    dependencies = doc["project"]["dependencies"] + sum(doc["project"]["optional-dependencies"].values(), [])
    assert not any(
        x.lower().split(">=")[0] in {"torch", "torchvision", "torchaudio", "transformer-engine", "deepspeed"}
        for x in dependencies
    )
