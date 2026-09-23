from dataclasses import dataclass
from .torch_runtime import torch, nn
from .checkpointing import activation_checkpoint
from .config import ShinraConfig
from .audit import parameter_ledger
from .blocks import ShinraBlock
from .normalization import RMSNorm
from .cache import GlobalKV, MemoryState, ShinraCache
from .losses import chunked_cross_entropy, selected_log_probs
from .multimodal import ShinraMultimodal
from .world import ShinraWorldModel
from .settings import ShinraRuntimeConfig, ShinraTrainingConfig, execution_segment_size


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
        self.multimodal = ShinraMultimodal(c, self.runtime)
        self.world = ShinraWorldModel(c, self.runtime)
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
        states = [current.layers.get(index) if current.length else None for index in range(len(self.layers))]
        width = execution_segment_size(policy, self.runtime, device_type=x.device.type)
        pieces = []
        for start in range(0, x.shape[1], width):
            piece, states = self._run_segment(x[:, start : start + width], states, current.length + start)
            pieces.append(piece)
        x = torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
        if updated is not None:
            updated.layers = {index: state for index, state in enumerate(states)}
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

    def _run_segment(self, hidden, states, offset):
        """Run every layer on one execution chunk, carrying recurrent and KV state.

        Checkpoint groups recompute layer ranges. They do not drop or detach state.
        Global layers still see the carried KV prefix, so the chunk is not a local window.
        """
        policy = self.training_config
        group = policy.checkpoint_group_size
        new_states = list(states)
        use_checkpoint = self.training and policy.gradient_checkpointing and torch.is_grad_enabled()
        te = bool(getattr(self, "_shinra_te_paths", ()))
        for start in range(0, len(self.layers), group):
            indices = tuple(range(start, min(start + group, len(self.layers))))
            packed, kinds = [], []
            for index in indices:
                first, second, kind = _pack_state(new_states[index], hidden)
                packed.extend((first, second))
                kinds.append(kind)
            out_kinds = tuple(self.layers[index].kind for index in indices)

            def run(value, *flat, indices=indices, kinds=tuple(kinds), offset=offset):
                local = [_unpack_state(kinds[i], flat[2 * i], flat[2 * i + 1]) for i in range(len(indices))]
                for slot, index in enumerate(indices):
                    value, local[slot] = self.layers[index](value, local[slot], offset)
                result = [value]
                for state in local:
                    first, second, _kind = _pack_state(state, value)
                    result.extend((first, second))
                return tuple(result)

            result = (
                activation_checkpoint(run, hidden, *packed, use_te=te)
                if use_checkpoint
                else run(hidden, *packed)
            )
            hidden = result[0]
            for slot, index in enumerate(indices):
                new_states[index] = _unpack_state(out_kinds[slot], result[1 + 2 * slot], result[2 + 2 * slot])
        return hidden, new_states

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
        *,
        cache=None,
        use_cache=False,
        previous_state=None,
    ):
        """observation + previous slots + action + Δt → next slots and all horizon heads.

        Training keeps the next state in the autograd graph. Inference cache stores a
        detached copy and never writes back into the source cache.
        """
        if previous_state is None and cache is not None:
            previous_state = cache.world.get("slots")
        slots = self.world.observe(observations, previous_state)
        conditioned = self.world.condition(slots, actions, action_mask, action_type, delta_t)
        embeds = self.multimodal.interfaces["state"].encode(conditioned)
        inference = not torch.is_grad_enabled() and (use_cache or cache is not None)
        result = self(inputs_embeds=embeds, cache=cache if inference else None, use_cache=inference)
        latent = self.multimodal.interfaces["state"].decode(result.hidden_states)
        if result.cache is not None:
            result.cache.world["slots"] = latent.detach()
        prediction = self.world.predict(latent, delta_t)
        prediction["next_state"] = latent
        return prediction, result.cache


def _pack_state(state, like):
    if state is None:
        return like.new_zeros(1), like.new_zeros(1), "none"
    if isinstance(state, MemoryState):
        return state.matrix, state.conv, "memory"
    if isinstance(state, GlobalKV):
        return state.key, state.value, "global"
    raise TypeError("Unknown backbone state")


def _unpack_state(kind, first, second):
    if kind == "none":
        return None
    if kind == "memory":
        return MemoryState(first, second)
    if kind == "global":
        return GlobalKV(first, second)
    raise TypeError("Unknown backbone state")
