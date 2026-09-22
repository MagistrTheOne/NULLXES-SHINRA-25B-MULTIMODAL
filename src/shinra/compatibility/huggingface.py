"""Optional HF export adapter, explicitly registered only by the caller."""

from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from ..config import ShinraConfig
from ..model import ShinraForCausalLM


class ShinraHFConfig(PretrainedConfig):
    model_type = "shinra"

    def __init__(self, shinra=None, **kwargs):
        super().__init__(**kwargs)
        self.shinra = shinra or ShinraConfig().to_dict()


class ShinraHFForCausalLM(PreTrainedModel):
    config_class = ShinraHFConfig
    base_model_prefix = "shinra"
    _supports_cache_class = False

    def __init__(self, config, *, runtime=None, training=None, allow_large_init=False):
        super().__init__(config)
        self.shinra = ShinraForCausalLM(
            ShinraConfig(**config.shinra),
            runtime=runtime,
            training=training,
            allow_large_init=allow_large_init,
        )
        # Do not call post_init: initialization/tied embeddings belong to the native model.

    def get_input_embeddings(self):
        return self.shinra.embed_tokens

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=1,
        attention_mask=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported HF arguments: {sorted(kwargs)}")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError(
                "Padded/packed episodes must be split explicitly; masks cannot leak across memory"
            )
        out = self.shinra(
            input_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            cache=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
        )
        return CausalLMOutputWithPast(loss=out.loss, logits=out.logits, past_key_values=out.cache)

    def generate(self, inputs=None, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs)
        return self.shinra.generate(input_ids, **kwargs)


def register():
    AutoConfig.register("shinra", ShinraHFConfig)
    AutoModelForCausalLM.register(ShinraHFConfig, ShinraHFForCausalLM)
