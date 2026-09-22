import torch
from torch import nn
from ..normalization import RMSNorm
from ..blocks import SwiGLU
from ..positions import rotary, spatial_rotary
from ..kernels import attention


class FrontendAttention(nn.Module):
    def __init__(self, dim, heads, eps=1e-6):
        super().__init__()
        self.heads, self.dim = heads, dim // heads
        self.q, self.k, self.v, self.o = [nn.Linear(dim, dim, bias=False) for _ in range(4)]
        self.q_norm, self.k_norm = RMSNorm(self.dim, eps), RMSNorm(self.dim, eps)

    def forward(
        self, x, source=None, positions=None, source_positions=None, causal=False, window=None, cache=None
    ):
        source = x if source is None else source
        b, s, _ = x.shape
        q = self.q_norm(self.q(x).reshape(b, s, self.heads, self.dim))
        k = self.k_norm(self.k(source).reshape(b, source.shape[1], self.heads, self.dim))
        v = self.v(source).reshape(b, source.shape[1], self.heads, self.dim)
        if positions is not None:
            source_positions = positions if source_positions is None else source_positions
            if positions.ndim == 3:
                q, k = spatial_rotary(q, positions), spatial_rotary(k, source_positions)
            else:
                q, k = rotary(q, positions, self.dim), rotary(k, source_positions, self.dim)
        offset = 0
        if cache is not None:
            offset = cache[0].shape[1]
            k, v = torch.cat((cache[0], k), 1), torch.cat((cache[1], v), 1)
        out = attention(q, k, v, causal=causal, offset=offset, window=window, reference_limit=8192)
        state = (k[:, -(window - 1) :].clone(), v[:, -(window - 1) :].clone()) if window else (k, v)
        return self.o(out.flatten(-2)), state


class FrontendBlock(nn.Module):
    def __init__(self, dim, ffn, heads, eps=1e-6):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.attn = FrontendAttention(dim, heads, eps)
        self.mlp = SwiGLU(dim, ffn)

    def forward(self, x, positions=None, causal=False, window=None, cache=None):
        y, state = self.attn(self.norm1(x), positions=positions, causal=causal, window=window, cache=cache)
        x = x + y
        return x + self.mlp(self.norm2(x)), state


class CrossBlock(nn.Module):
    def __init__(self, dim, ffn, heads, eps=1e-6):
        super().__init__()
        self.query_norm, self.source_norm, self.ffn_norm = [RMSNorm(dim, eps) for _ in range(3)]
        self.attn = FrontendAttention(dim, heads, eps)
        self.mlp = SwiGLU(dim, ffn)

    def forward(self, queries, source):
        y, _ = self.attn(self.query_norm(queries), self.source_norm(source))
        x = queries + y
        return x + self.mlp(self.ffn_norm(x))


class Resampler(nn.Module):
    def __init__(self, dim, ffn, heads, layers, queries, eps=1e-6):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(queries, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList([CrossBlock(dim, ffn, heads, eps) for _ in range(layers)])
        self.norm = RMSNorm(dim, eps)

    def forward(self, source, count, query_offset=0):
        if count < 1 or count + query_offset > len(self.queries):
            raise ValueError("Resampler query budget exceeded")
        x = self.queries[query_offset : query_offset + count].unsqueeze(0).expand(source.shape[0], -1, -1)
        for layer in self.layers:
            x = layer(x, source)
        return self.norm(x)
