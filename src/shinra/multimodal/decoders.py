import torch
from torch import nn
from .layers import FrontendBlock
from .image import patchify, unpatchify
from ..normalization import RMSNorm
from ..positions import fourier_features


class VisualFlowDecoder(nn.Module):
    def __init__(self, c, runtime=None):
        super().__init__()
        self.config = c
        d, p = c.latent_dim, 3 * c.patch_size**2
        self.input, self.output = nn.Linear(p, d, bias=False), nn.Linear(d, p, bias=False)
        self.condition = nn.Linear(d, d, bias=False)
        self.time = nn.Linear(c.time_features, d, bias=False)
        self.layers = nn.ModuleList(
            [
                FrontendBlock(
                    d,
                    c.latent_ffn,
                    c.latent_heads,
                    c.norm_eps,
                    runtime=runtime,
                    rope_theta=c.rope_theta,
                    spatial_rope_theta=c.spatial_rope_theta,
                )
                for _ in range(c.visual_decoder_layers)
            ]
        )
        self.norm = RMSNorm(d, c.norm_eps)

    def forward(self, noisy, condition, time):
        c = self.config
        tokens = patchify(noisy, c.patch_size)
        if tokens.shape[1] > c.max_visual_patches:
            raise ValueError("Decoder patch budget exceeded")
        n = tokens.shape[1]
        x = torch.cat((self.condition(condition), self.input(tokens)), 1)
        x = x + self.time(fourier_features(time[:, None], c.time_features).to(x.dtype))
        rows, cols = noisy.shape[-2] // c.patch_size, noisy.shape[-1] // c.patch_size
        yy, xx = torch.meshgrid(
            torch.arange(rows, device=x.device), torch.arange(cols, device=x.device), indexing="ij"
        )
        # Both streams have explicit positions. Latent queries are ordered resampler slots.
        patch_xy = torch.stack((xx, yy), -1).reshape(1, n, 2).expand(x.shape[0], -1, -1)
        index = torch.arange(condition.shape[1], device=x.device, dtype=torch.float32)
        cond_xy = torch.stack((index % cols, torch.div(index, cols, rounding_mode="floor")), -1)
        xy = torch.cat((cond_xy[None].expand(x.shape[0], -1, -1), patch_xy), 1)
        for layer in self.layers:
            x, _ = layer(x, positions=xy)
        return unpatchify(self.output(self.norm(x[:, -n:])), noisy.shape[-2], noisy.shape[-1], c.patch_size)


class AudioFlowDecoder(nn.Module):
    def __init__(self, c, runtime=None):
        super().__init__()
        self.config = c
        d, p = c.audio_decoder_dim, c.audio_output_patch
        self.input, self.output = nn.Linear(p, d, bias=False), nn.Linear(d, p, bias=False)
        self.condition = nn.Linear(c.latent_dim, c.audio_decoder_expand * d, bias=False)
        self.time = nn.Linear(c.time_features, d, bias=False)
        self.layers = nn.ModuleList(
            [
                FrontendBlock(
                    d,
                    c.audio_decoder_ffn,
                    c.audio_decoder_heads,
                    c.norm_eps,
                    runtime=runtime,
                    rope_theta=c.audio_rope_theta,
                    spatial_rope_theta=c.spatial_rope_theta,
                )
                for _ in range(c.audio_decoder_layers)
            ]
        )
        self.norm = RMSNorm(d, c.norm_eps)

    def forward(self, noisy, condition, time):
        c = self.config
        expand = c.audio_decoder_expand
        if noisy.shape[1] != condition.shape[1] * expand * c.audio_output_patch:
            raise ValueError("Audio output must contain the configured waveform patches per latent")
        patches = noisy.reshape(noisy.shape[0], -1, c.audio_output_patch)
        x = self.input(patches) + self.condition(condition).reshape(noisy.shape[0], -1, c.audio_decoder_dim)
        x = x + self.time(fourier_features(time[:, None], c.time_features).to(x.dtype))
        positions = torch.arange(x.shape[1], device=x.device)
        for layer in self.layers:
            x, _ = layer(x, positions=positions, window=c.audio_window)
        return self.output(self.norm(x)).flatten(1)


def flow_matching_loss(decoder, target, condition, generator=None):
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    t = torch.rand(target.shape[0], device=target.device, generator=generator)
    broadcast = t.reshape(-1, *([1] * (target.ndim - 1)))
    mixed = (1 - broadcast) * noise + broadcast * target
    predicted = decoder(mixed.to(target.dtype), condition, t)
    return (predicted.float() - (target - noise).float()).square().mean()


@torch.no_grad()
def sample_flow(decoder, noise, condition, steps=32):
    if steps < 1:
        raise ValueError("Flow steps must be positive")
    x = noise
    for index in range(steps):
        t = torch.full((x.shape[0],), index / steps, device=x.device)
        velocity = decoder(x, condition, t)
        midpoint = x + velocity * (0.5 / steps)
        x = x + decoder(midpoint, condition, t + 0.5 / steps) / steps
    return x
