import math
import torch


def fourier_features(values, width):
    """Physical coordinates -> fixed Fourier features, no trainable lookup limit."""
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    axes = values.shape[-1]
    pairs = math.ceil(width / (2 * axes))
    freq = torch.exp(torch.linspace(0, math.log(10000), pairs, device=values.device))
    phase = values.float().unsqueeze(-1) / freq
    return torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)[..., :width]


def rotary(x, positions, dim, theta=1e6):
    """x: B,S,H,D. positions: S or B,S. No growing cosine cache."""
    inv = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    phase = positions.float().unsqueeze(-1) * inv
    if phase.ndim == 2:
        phase = phase.unsqueeze(0)
    cos, sin = phase.cos().unsqueeze(2).to(x.dtype), phase.sin().unsqueeze(2).to(x.dtype)
    a, b = x[..., :dim:2], x[..., 1:dim:2]
    part = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    return torch.cat((part, x[..., dim:]), dim=-1)


def spatial_rotary(x, xy, theta=10000.0):
    half = x.shape[-1] // 2
    return torch.cat(
        (rotary(x[..., :half], xy[..., 0], half, theta), rotary(x[..., half:], xy[..., 1], half, theta)),
        dim=-1,
    )
