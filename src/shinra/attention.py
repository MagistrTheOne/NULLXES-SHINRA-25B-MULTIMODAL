from torch import nn
import torch
from .normalization import RMSNorm
from .positions import rotary
from .kernels import attention
from .cache import GlobalKV
from .settings import ShinraRuntimeConfig


class ShinraGlobalAttention(nn.Module):
    def __init__(self, c, runtime=None):
        super().__init__()
        self.config = c
        self.runtime = runtime or ShinraRuntimeConfig()
        h, kv = c.hidden_size, c.num_key_value_heads * c.head_dim
        self.q = nn.Linear(h, h, bias=False)
        self.k = nn.Linear(h, kv, bias=False)
        self.v = nn.Linear(h, kv, bias=False)
        self.o = nn.Linear(h, h, bias=False)
        self.q_norm, self.k_norm = RMSNorm(c.head_dim, c.norm_eps), RMSNorm(c.head_dim, c.norm_eps)

    def forward(self, x, state=None, offset=0):
        c = self.config
        b, s, _ = x.shape
        q = self.q_norm(self.q(x).view(b, s, c.num_attention_heads, c.head_dim))
        k = self.k_norm(self.k(x).view(b, s, c.num_key_value_heads, c.head_dim))
        v = self.v(x).view(b, s, c.num_key_value_heads, c.head_dim)
        pos = torch.arange(offset, offset + s, device=x.device)
        q, k = rotary(q, pos, c.rotary_dim, c.rope_theta), rotary(k, pos, c.rotary_dim, c.rope_theta)
        if state is not None:
            k, v = torch.cat((state.key, k), dim=1), torch.cat((state.value, v), dim=1)
        y = attention(
            q,
            k,
            v,
            backend=self.runtime.attention_backend,
            offset=offset,
            reference_limit=self.runtime.reference_backend_max_tokens,
        )
        return self.o(y.flatten(-2)), GlobalKV(k, v)
