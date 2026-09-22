"""Chunked exact losses: never retain [B,S,V] logits for backward."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_cross_entropy(hidden, weight, labels, *, chunk_size=256, z_loss=0.0, ignore_index=-100):
    if labels.shape != hidden.shape[:-1]:
        raise ValueError("Targets must align with prediction positions; shift labels at the caller")
    flat, targets = hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1)
    count = (targets != ignore_index).sum()

    def part_loss(x, y):
        logits = F.linear(x, weight).float()
        valid = y != ignore_index
        ce = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="sum")
        if z_loss:
            ce = ce + z_loss * (torch.logsumexp(logits, dim=-1).square() * valid).sum()
        return ce

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk_size):
        x, y = flat[start : start + chunk_size], targets[start : start + chunk_size]
        value = (
            checkpoint(part_loss, x, y, use_reentrant=False) if torch.is_grad_enabled() else part_loss(x, y)
        )
        total = total + value
    return total / count.clamp_min(1), count


def selected_log_probs(hidden, weight, targets, chunk_size=256):
    flat, ids = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)

    def run(x, y):
        logits = F.linear(x, weight).float()
        return logits.gather(1, y[:, None]).squeeze(1) - torch.logsumexp(logits, dim=-1)

    outputs = []
    for start in range(0, len(ids), chunk_size):
        args = (flat[start : start + chunk_size], ids[start : start + chunk_size])
        outputs.append(checkpoint(run, *args, use_reentrant=False) if torch.is_grad_enabled() else run(*args))
    return torch.cat(outputs).reshape(targets.shape)


def gaussian_nll(mean, log_variance, target, mask=None):
    log_variance = log_variance.float().clamp(-12, 8)
    loss = 0.5 * ((target.float() - mean.float()).square() * (-log_variance).exp() + log_variance)
    if mask is None:
        return loss.mean()
    mask = torch.broadcast_to(mask, loss.shape)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def future_state_loss(mean, log_variance, target, mask=None):
    # Target encoder receives reconstruction/variance supervision, not a collapsing prediction target.
    return gaussian_nll(mean, log_variance, target.detach(), mask)


def latent_regularization(latents, variance_floor=1.0):
    x = latents.float().reshape(-1, latents.shape[-1])
    if x.shape[0] < 2:
        raise ValueError("Variance regularization needs at least two latent samples")
    centered = x - x.mean(0)
    cov = centered.T @ centered / (x.shape[0] - 1)
    variance = F.relu(variance_floor - torch.sqrt(cov.diag() + 1e-4)).mean()
    offdiag = cov - torch.diag_embed(cov.diag())
    return variance + offdiag.square().sum() / x.shape[1]
