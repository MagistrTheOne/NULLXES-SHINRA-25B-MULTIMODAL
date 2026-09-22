from dataclasses import dataclass
from .torch_runtime import torch, nn
from .checkpointing import activation_checkpoint
from .config import ShinraConfig
from .audit import parameter_ledger
from .blocks import ShinraBlock
from .normalization import RMSNorm
from .cache import ShinraCache
from .losses import chunked_cross_entropy, selected_log_probs
from .multimodal import ShinraMultimodal
from .world import ShinraWorldModel
from .settings import ShinraRuntimeConfig, ShinraTrainingConfig


@dataclass
class ShinraOutput:
    hidden_states: torch.Tensor
    loss: torch.Tensor | None = None
    valid_targets: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    cache: ShinraCache | None = None


class ShinraForCausalLM(nn.Module):
    """Single native family. Large allocation requires an explicit opt-in."""

    def __init__(self, config=None, *, runtime=None, training=None, allow_large_init=False):
        super().__init__()
        self.config = config or ShinraConfig()
        self.runtime = runtime or ShinraRuntimeConfig()
        self.training_config = training or ShinraTrainingConfig()
        c = self.config
        if parameter_ledger(c)["total"] > 100_000_000 and not allow_large_init:
            raise RuntimeError("Large model allocation requires allow_large_init=True; audit needs no model")
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(
            [ShinraBlock(c, kind, self.runtime, self.training_config) for kind in c.layer_types]
        )
        self.norm = RMSNorm(c.hidden_size, c.norm_eps)
        self.multimodal = ShinraMultimodal(c)
        self.world = ShinraWorldModel(c)
        self.reset_parameters()

    @property
    def lm_head_weight(self):
        return self.embed_tokens.weight

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=module.in_features**-0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        with torch.no_grad():
            scale = (2 * self.config.num_hidden_layers) ** -0.5
            for block in self.layers:
                block.mixer.o.weight.mul_(scale)
                block.mlp.down.weight.mul_(scale)
                if block.kind == "memory":
                    block.mixer.reset_gates()

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        labels=None,
        cache=None,
        use_cache=False,
        logits_to_keep=0,
    ):
        c = self.config
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds")
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if x.ndim != 3 or x.shape[-1] != c.hidden_size or x.shape[1] < 1:
            raise ValueError("Invalid backbone input shape")
        if cache is not None and not use_cache:
            raise ValueError("Passing cache requires use_cache=True")
        if use_cache and torch.is_grad_enabled():
            raise ValueError("Persistent cache is inference-only; training uses differentiable recurrence")
        if cache is not None and labels is not None:
            raise ValueError("Cached training targets are not supported")
        current = cache or ShinraCache(c.fingerprint(), c.cache_version)
        current.validate(c, x.shape[1])
        updated = current.fork() if use_cache else None
        policy = self.training_config
        if policy.gradient_checkpointing and self.training and torch.is_grad_enabled():
            for start in range(0, len(self.layers), policy.checkpoint_group_size):
                group = tuple(self.layers[start : start + policy.checkpoint_group_size])

                def run_group(value, blocks=group):
                    for block in blocks:

                        def run_layer(inner, current_block=block):
                            return current_block(inner)[0]

                        value = activation_checkpoint(
                            run_layer, value, use_te=bool(getattr(self, "_shinra_te_paths", ()))
                        )
                    return value

                x = activation_checkpoint(run_group, x, use_te=bool(getattr(self, "_shinra_te_paths", ())))
        else:
            for index, layer in enumerate(self.layers):
                state = current.layers.get(index)
                x, new_state = layer(x, state, current.length)
                if use_cache:
                    updated.layers[index] = new_state
        x = self.norm(x)
        if updated is not None:
            updated.length += x.shape[1]
        loss, count = None, None
        if labels is not None:
            if labels.shape != x.shape[:2] or x.shape[1] < 2:
                raise ValueError("Causal labels must match the sequence and contain at least two positions")
            loss, count = chunked_cross_entropy(
                x[:, :-1],
                self.lm_head_weight,
                labels[:, 1:],
                chunk_size=policy.loss_chunk_size,
                z_loss=policy.z_loss,
            )
        if not 0 <= logits_to_keep <= policy.loss_chunk_size:
            raise ValueError("Logits output is bounded; use iter_logits for larger requests")
        logits = nn.functional.linear(x[:, -logits_to_keep:], self.lm_head_weight) if logits_to_keep else None
        return ShinraOutput(x, loss, count, logits, updated)

    def iter_logits(self, hidden):
        for part in hidden.split(self.training_config.loss_chunk_size, dim=1):
            yield nn.functional.linear(part, self.lm_head_weight)

    def log_probs(self, hidden, targets):
        return selected_log_probs(hidden, self.lm_head_weight, targets, self.training_config.loss_chunk_size)

    @torch.no_grad()
    def prefill(self, input_ids=None, *, inputs_embeds=None, cache=None, chunk_size=1024):
        if chunk_size < 1:
            raise ValueError("Prefill chunk_size must be positive")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one input representation")
        source = input_ids if input_ids is not None else inputs_embeds
        if source.shape[1] < 1:
            raise ValueError("Empty prefill")
        output = None
        for chunk in source.split(chunk_size, dim=1):
            kwargs = {"input_ids": chunk} if input_ids is not None else {"inputs_embeds": chunk}
            output = self(**kwargs, cache=cache, use_cache=True, logits_to_keep=1)
            cache = output.cache
        return output

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=32, eos_token_id=None, temperature=0.0, generator=None):
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("Invalid generation budget or temperature")
        if input_ids.shape[0] != 1:
            raise ValueError("Generation supports one episode per call; batch scheduling is external")
        if input_ids.shape[1] + max_new_tokens > self.config.max_context_length:
            raise ValueError("Prompt plus generation exceeds native context")
        if max_new_tokens == 0:
            return input_ids
        out = self.prefill(input_ids)
        tokens = [input_ids]
        for step in range(max_new_tokens):
            logits = out.logits[:, -1].float()
            token = (
                logits.argmax(-1, keepdim=True)
                if temperature == 0
                else torch.multinomial((logits / temperature).softmax(-1), 1, generator=generator)
            )
            tokens.append(token)
            if eos_token_id is not None and token.item() == eos_token_id:
                break
            if step + 1 < max_new_tokens:
                out = self(token, cache=out.cache, use_cache=True, logits_to_keep=1)
        return torch.cat(tokens, 1)

    def predict_world(
        self,
        observations,
        actions,
        action_mask,
        action_type,
        delta_t,
        horizon,
        *,
        cache=None,
        use_cache=False,
    ):
        slots = self.world.observe(observations)
        conditioned = self.world.condition(slots, actions, action_mask, action_type, delta_t, horizon)
        embeds = self.multimodal.interfaces["state"].encode(conditioned)
        result = self(inputs_embeds=embeds, cache=cache, use_cache=use_cache or cache is not None)
        latent = self.multimodal.interfaces["state"].decode(result.hidden_states)
        if result.cache is not None:
            result.cache.world["slots"] = latent.detach()
        return self.world.predict(latent), result.cache
