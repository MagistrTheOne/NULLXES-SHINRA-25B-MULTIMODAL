import torch
from torch import nn
import torch.nn.functional as F
from .multimodal.layers import Resampler
from .positions import fourier_features


class ShinraWorldModel(nn.Module):
    """Small heads around shared-backbone dynamics, not an independent dynamics network."""

    def __init__(self, c):
        super().__init__()
        self.config = c
        d = c.latent_dim
        self.slots = Resampler(
            d, c.latent_ffn, c.latent_heads, c.world_slot_layers, c.world_slots, c.norm_eps
        )
        self.future_up = nn.Linear(d, c.latent_ffn, bias=False)
        self.future_down = nn.Linear(c.latent_ffn, 2 * d, bias=False)
        self.action_in = nn.Linear(2 * c.action_dim, d, bias=False)
        self.action_type = nn.Embedding(c.action_types, d)
        self.action_out = nn.Linear(d, 2 * c.action_dim, bias=False)
        self.time = nn.Linear(c.time_features, d, bias=False)
        self.horizon = nn.Embedding(c.world_horizons, d)
        self.events = nn.Linear(d, c.world_events, bias=False)
        self.reward_terminal = nn.Linear(d, 4, bias=False)

    def observe(self, latents):
        return self.slots(latents, self.config.world_slots)

    def condition(self, slots, actions, action_mask, action_type, delta_t, horizon):
        c = self.config
        if actions.shape != action_mask.shape or actions.shape[-1] != c.action_dim:
            raise ValueError("Action values and masks must match the configured action dimension")
        if (delta_t < 0).any():
            raise ValueError("Future prediction requires nonnegative delta_t")
        encoded = self.action_in(torch.cat((actions * action_mask, action_mask.to(actions.dtype)), -1))
        encoded = encoded + self.action_type(action_type)
        timing = self.time(fourier_features(delta_t.reshape(-1, 1), c.time_features).to(slots.dtype)).squeeze(
            1
        )
        return slots + (encoded + timing + self.horizon(horizon))[:, None, :]

    def predict(self, backbone_latents):
        mean, log_variance = self.future_down(F.silu(self.future_up(backbone_latents))).chunk(2, -1)
        action_mean, action_log_variance = self.action_out(backbone_latents).chunk(2, -1)
        return {
            "mean": mean,
            "log_variance": log_variance.clamp(-12, 8),
            "action_mean": action_mean,
            "action_log_variance": action_log_variance.clamp(-12, 8),
            "events": self.events(backbone_latents),
            "reward_terminal": self.reward_terminal(backbone_latents),
        }
