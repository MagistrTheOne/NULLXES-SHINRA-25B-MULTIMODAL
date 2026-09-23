import torch
from torch import nn
import torch.nn.functional as F
from .multimodal.layers import Resampler
from .positions import fourier_features
from .settings import ShinraRuntimeConfig


class ShinraWorldModel(nn.Module):
    """Slot state plus compact multi-horizon heads. Not an autoregressive simulator.

    Transition inputs are the current observations, the previous slot state when an
    episode is already open, the action, and a base delta_t. Horizon heads then read
    that updated state at every configured multiple of delta_t.
    """

    def __init__(self, c, runtime=None):
        super().__init__()
        self.config = c
        self.runtime = runtime or ShinraRuntimeConfig()
        d = c.latent_dim
        self.slots = Resampler(
            d,
            c.latent_ffn,
            c.latent_heads,
            c.world_slot_layers,
            c.world_slots,
            c.norm_eps,
            runtime=self.runtime,
        )
        self.future_up = nn.Linear(d, c.latent_ffn, bias=False)
        self.future_down = nn.Linear(c.latent_ffn, 2 * d, bias=False)
        self.action_in = nn.Linear(2 * c.action_dim, d, bias=False)
        self.action_type = nn.Embedding(c.action_types, d)
        self.action_out = nn.Linear(d, 2 * c.action_dim, bias=False)
        self.time = nn.Linear(c.time_features, d, bias=False)
        self.horizon = nn.Embedding(c.world_horizons, d)
        self.events = nn.Linear(d, c.world_events, bias=False)
        # Fixed channel order is config.world_aux_channels: reward, terminal, value, cost.
        self.reward_terminal = nn.Linear(d, len(c.world_aux_channels), bias=False)

    def observe(self, latents, previous=None):
        """Pool observations. An existing slot tensor is part of the source, not a label."""
        if latents.ndim != 3:
            raise ValueError("Observations must have shape B, N, D")
        source = latents
        if previous is not None:
            if previous.shape[0] != latents.shape[0] or previous.shape[-1] != latents.shape[-1]:
                raise ValueError("Previous world state must match the observation batch and width")
            source = torch.cat((previous, latents), dim=1)
        return self.slots(source, self.config.world_slots)

    def condition(self, slots, actions, action_mask, action_type, delta_t):
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
        return slots + encoded[:, None, :] + timing[:, None, :]

    def predict(self, state, delta_t):
        c = self.config
        batch, _slots, _dim = state.shape
        steps = torch.tensor(c.world_horizon_steps, device=state.device, dtype=state.dtype)
        times = delta_t.reshape(batch, 1).to(dtype=state.dtype) * steps.reshape(1, -1)
        timing = self.time(fourier_features(times, c.time_features).to(state.dtype))
        bias = self.horizon.weight
        expanded = state[:, :, None, :] + timing[:, None, :, :] + bias[None, None, :, :]
        flat = expanded.reshape(batch, -1, state.shape[-1])
        mean, log_variance = self.future_down(F.silu(self.future_up(flat))).chunk(2, -1)
        horizons = len(c.world_horizon_steps)
        mean = mean.view(batch, _slots, horizons, -1)
        log_variance = log_variance.view(batch, _slots, horizons, -1).clamp(-12, 8)
        action_mean, action_log_variance = self.action_out(state).chunk(2, -1)
        aux = self.reward_terminal(state)
        return {
            "mean": mean,
            "log_variance": log_variance,
            "horizon_times": times,
            "action_mean": action_mean,
            "action_log_variance": action_log_variance.clamp(-12, 8),
            "events": self.events(state),
            "aux": {name: aux[..., index] for index, name in enumerate(c.world_aux_channels)},
        }
