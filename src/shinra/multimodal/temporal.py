from torch import nn
from .layers import FrontendBlock
from ..normalization import RMSNorm
from ..positions import fourier_features


class ShinraTemporalModule(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.features = c.time_features
        self.time = nn.Linear(c.time_features, c.latent_dim, bias=False)
        self.layers = nn.ModuleList(
            [
                FrontendBlock(c.latent_dim, c.latent_ffn, c.latent_heads, c.norm_eps)
                for _ in range(c.temporal_layers)
            ]
        )
        self.norm = RMSNorm(c.latent_dim, c.norm_eps)

    def forward(self, x, times):
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(x.shape[0], -1)
        x = x + self.time(fourier_features(times, self.features).to(x.dtype))
        for layer in self.layers:
            x, _ = layer(x, positions=times, causal=True)
        return self.norm(x)
