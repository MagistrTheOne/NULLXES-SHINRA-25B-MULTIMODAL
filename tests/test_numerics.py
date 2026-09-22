from dataclasses import replace
import torch
import torch.nn.functional as F
from shinra.kernels import delta_reference
from shinra.losses import chunked_cross_entropy, selected_log_probs
from shinra.model import ShinraForCausalLM
from shinra.cache import ShinraCache


def test_recurrence_gradcheck_and_split():
    args = [torch.randn(1, 4, 2, 3, dtype=torch.double, requires_grad=True) for _ in range(3)]
    decay = (-torch.rand(1, 4, 2, 3, dtype=torch.double)).requires_grad_()
    beta = torch.rand(1, 4, 2, dtype=torch.double, requires_grad=True)
    state = torch.randn(1, 2, 3, 3, dtype=torch.double, requires_grad=True)
    all_args = (*args, decay, beta, state)
    assert torch.autograd.gradcheck(delta_reference, all_args, fast_mode=True)
    full, end = delta_reference(*all_args)
    first, middle = delta_reference(*(x[:, :2] for x in all_args[:-1]), state)
    second, last = delta_reference(*(x[:, 2:] for x in all_args[:-1]), middle)
    torch.testing.assert_close(full, torch.cat((first, second), 1))
    torch.testing.assert_close(end, last)


def test_chunked_ce_gradients_and_mask():
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    weight = torch.randn(19, 8, requires_grad=True)
    targets = torch.randint(0, 19, (2, 7))
    targets[0, 3] = -100
    loss, count = chunked_cross_entropy(hidden, weight, targets, chunk_size=3, z_loss=0.01)
    grads = torch.autograd.grad(loss, (hidden, weight))
    logits = F.linear(hidden, weight).float()
    ref = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
    ref += 0.01 * (logits.logsumexp(-1).square() * (targets != -100)).sum() / count
    ref_grads = torch.autograd.grad(ref, (hidden, weight))
    torch.testing.assert_close(loss, ref)
    for actual, expected in zip(grads, ref_grads):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    ignored, _ = chunked_cross_entropy(hidden, weight, torch.full_like(targets, -100), chunk_size=3)
    assert ignored.item() == 0


def test_selected_log_probs_exact():
    h, w = torch.randn(2, 5, 8), torch.randn(19, 8)
    y = torch.randint(0, 19, (2, 5))
    torch.testing.assert_close(
        selected_log_probs(h, w, y, 3), F.linear(h, w).log_softmax(-1).gather(-1, y[..., None]).squeeze(-1)
    )


def test_cache_prefill_roundtrip_and_branch(config, tmp_path):
    model = ShinraForCausalLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 13))
    with torch.no_grad():
        full = model(ids).hidden_states
        initial = model(ids[:, :7], use_cache=True)
        initial.cache.save(tmp_path / "state")
        restored = ShinraCache.load(tmp_path / "state", config)
        tail = model(ids[:, 7:], cache=restored, use_cache=True)
        torch.testing.assert_close(tail.hidden_states, full[:, 7:], rtol=2e-4, atol=2e-5)
        assert initial.cache.length == 7 and restored.length == 7 and tail.cache.length == 13
        chunked = model.prefill(ids, chunk_size=3)
        torch.testing.assert_close(chunked.hidden_states[:, -1], full[:, -1], rtol=2e-4, atol=2e-5)
        assert restored.reset("next").length == 0


def test_checkpoint_gradient_equivalence(config):
    base = ShinraForCausalLM(config)
    checked = ShinraForCausalLM(replace(config, gradient_checkpointing=True))
    checked.load_state_dict(base.state_dict())
    tokens = torch.randint(0, config.vocab_size, (1, 9))
    base(tokens, labels=tokens).loss.backward()
    checked(tokens, labels=tokens).loss.backward()
    for (name, p), (other, q) in zip(base.named_parameters(), checked.named_parameters()):
        assert name == other
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-6, msg=name)
