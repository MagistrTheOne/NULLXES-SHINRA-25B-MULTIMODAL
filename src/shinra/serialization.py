"""Sharded safetensors, strict manifests and atomic directory publication."""

import hashlib
import json
import os
from pathlib import Path
import uuid
import torch


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(model, directory, *, optimizer=None, training_state=None, shard_bytes=2_000_000_000):
    from safetensors.torch import save_file

    destination = Path(directory)
    if destination.exists():
        raise FileExistsError("Checkpoint publication never overwrites existing checkpoints")
    if getattr(model, "_shinra_te_paths", ()):
        raise ValueError("TE training state requires distributed checkpoint state_dict serialization")
    if shard_bytes <= 0:
        raise ValueError("shard_bytes must be positive")
    temporary = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    model.config.save(temporary / "config.json")
    model.runtime.save(temporary / "runtime.json")
    model.training_config.save(temporary / "training.json")
    index, hashes, shard = {}, {}, {}
    size, number = 0, 0

    def flush():
        nonlocal shard, size, number
        if not shard:
            return
        name = f"weights-{number:05d}.safetensors"
        save_file(shard, str(temporary / name))
        hashes[name] = _sha256(temporary / name)
        index.update({key: name for key in shard})
        shard, size, number = {}, 0, number + 1

    # named_parameters excludes aliases, and the LM head has no duplicate parameter.
    for name, tensor in model.named_parameters():
        if hasattr(tensor, "to_local"):
            raise ValueError("Use distributed checkpointing for sharded tensors")
        nbytes = tensor.numel() * tensor.element_size()
        if shard and size + nbytes > shard_bytes:
            flush()
        shard[name] = tensor.detach().cpu().contiguous()
        size += nbytes
    flush()
    state = dict(training_state or {})
    (temporary / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        hashes["optimizer.pt"] = _sha256(temporary / "optimizer.pt")
    rng = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, temporary / "rng.pt")
    hashes["rng.pt"] = _sha256(temporary / "rng.pt")
    for name in ("config.json", "runtime.json", "training.json", "trainer_state.json"):
        hashes[name] = _sha256(temporary / name)
    manifest = {
        "format": "shinra-safetensors-v2",
        "fingerprint": model.config.fingerprint(),
        "weight_map": index,
        "sha256": hashes,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def load_checkpoint(model, directory, *, optimizer=None, restore_rng=False, verify_hashes=True):
    from safetensors import safe_open

    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "shinra-safetensors-v2" or manifest["fingerprint"] != model.config.fingerprint():
        raise ValueError("Incompatible checkpoint contract")

    def checked(name):
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError("Invalid checkpoint manifest path")
        return path

    if verify_hashes:
        for name, digest in manifest["sha256"].items():
            if _sha256(checked(name)) != digest:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
    params = dict(model.named_parameters())
    if set(params) != set(manifest["weight_map"]):
        raise ValueError("Checkpoint parameter keys differ from model")
    # Validate every shape before copying any weight.
    for name in set(manifest["weight_map"].values()):
        with safe_open(checked(name), framework="pt", device="cpu") as handle:
            expected = {key for key, filename in manifest["weight_map"].items() if filename == name}
            if set(handle.keys()) != expected:
                raise ValueError("Shard keys differ from checkpoint manifest")
            for key in handle.keys():
                if key not in params or tuple(handle.get_slice(key).get_shape()) != tuple(params[key].shape):
                    raise ValueError(f"Checkpoint shape mismatch: {key}")
    with torch.no_grad():
        for name in sorted(set(manifest["weight_map"].values())):
            with safe_open(checked(name), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    params[key].copy_(handle.get_tensor(key))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(root / "optimizer.pt", map_location="cpu", weights_only=True))
    if restore_rng:
        rng = torch.load(root / "rng.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["cpu"])
        if "cuda" in rng:
            if not torch.cuda.is_initialized():
                raise RuntimeError("CUDA RNG restoration requires an initialized matching runtime")
            torch.cuda.set_rng_state_all(rng["cuda"])
    return json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
