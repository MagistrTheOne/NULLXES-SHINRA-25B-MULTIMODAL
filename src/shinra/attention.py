from torch import nn
import torch
from .normalization import RMSNorm
from .positions import rotary
from .kernels import attention
from .cache import GlobalKV
from .settings import ShinraRuntimeConfig, ShinraTrainingConfig, execution_segment_size


class ShinraGlobalAttention(nn.Module):
    """Causal attention over the whole episode. Query chunks do not make it local."""

    def __init__(self, c, runtime=None, training=None):
        super().__init__()
        self.config = c
        self.runtime = runtime or ShinraRuntimeConfig()
        self.training_config = training or ShinraTrainingConfig()
        h, kv = c.hidden_size, c.num_key_value_heads * c.head_dim
        self.q = nn.Linear(h, h, bias=False)
        self.k = nn.Linear(h, kv, bias=False)
        self.v = nn.Linear(h, kv, bias=False)
        self.o = nn.Linear(h, h, bias=False)
        self.q_norm, self.k_norm = RMSNorm(c.head_dim, c.norm_eps), RMSNorm(c.head_dim, c.norm_eps)

    def forward(self, x, state=None, offset=0):
        c = self.config
        b, s, _ = x.shape
        width = execution_segment_size(self.training_config, self.runtime, device_type=x.device.type)
        keys = None if state is None else state.key
        values = None if state is None else state.value
        outputs = []
        for start in range(0, s, width):
            xs = x[:, start : start + width]
            q = self.q_norm(self.q(xs).view(b, xs.shape[1], c.num_attention_heads, c.head_dim))
            k = self.k_norm(self.k(xs).view(b, xs.shape[1], c.num_key_value_heads, c.head_dim))
            v = self.v(xs).view(b, xs.shape[1], c.num_key_value_heads, c.head_dim)
            pos = torch.arange(offset + start, offset + start + xs.shape[1], device=x.device)
            q = rotary(q, pos, c.rotary_dim, c.rope_theta)
            k = rotary(k, pos, c.rotary_dim, c.rope_theta)
            keys = k if keys is None else torch.cat((keys, k), dim=1)
            values = v if values is None else torch.cat((values, v), dim=1)
            # offset is the key count before this query chunk, so causality stays global.
            y = attention(
                q,
                keys,
                values,
                backend=self.runtime.attention_backend,
                offset=offset + start,
                reference_limit=self.runtime.reference_backend_max_tokens,
            )
            outputs.append(y)
        return self.o(torch.cat(outputs, dim=1).flatten(-2)), GlobalKV(keys, values)
