from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import T5EncoderModel

import types

import torch
import torch.nn as nn
from diffusers import FluxPipeline
from einops import rearrange
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_transformer
from nunchaku.caching.utils import cache_context, create_cache_context
from nunchaku.lora.flux.compose import compose_lora
from nunchaku.utils import load_state_dict_in_safetensors

from backend.utils import pad_to_patch_size
from modules.shared import opts


class SVDQFluxTransformer2DModel(nn.Module):
    """https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/v0.3.3/wrappers/flux.py#L14"""

    def __init__(self, config: dict, path: str):
        super(SVDQFluxTransformer2DModel, self).__init__()
        model = NunchakuFluxTransformer2dModel.from_pretrained(path, offload=opts.svdq_cpu_offload)
        model = apply_cache_on_transformer(transformer=model, residual_diff_threshold=opts.svdq_cache_threshold)
        model.set_attention_impl(opts.svdq_attention)

        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        self.loras = []

        # for first-block cache
        self._prev_timestep = None
        self._cache_context = None

    def forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            assert isinstance(timestep, float)
            timestep_float = timestep

        model = self.model
        assert isinstance(model, NunchakuFluxTransformer2dModel)

        bs, c, h, w = x.shape
        patch_size = self.config["patch_size"]
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size
        img_ids = FluxPipeline._prepare_latent_image_ids(bs, h_len, w_len, x.device, x.dtype)
        txt_ids = torch.zeros((context.shape[1], 3), device=x.device, dtype=x.dtype)

        # load and compose LoRA
        if self.loras != model.comfy_lora_meta_list:
            lora_to_be_composed = []
            for _ in range(max(0, len(model.comfy_lora_meta_list) - len(self.loras))):
                model.comfy_lora_meta_list.pop()
                model.comfy_lora_sd_list.pop()
            for i in range(len(self.loras)):
                meta = self.loras[i]
                if i >= len(model.comfy_lora_meta_list):
                    sd = load_state_dict_in_safetensors(meta[0])
                    model.comfy_lora_meta_list.append(meta)
                    model.comfy_lora_sd_list.append(sd)
                elif model.comfy_lora_meta_list[i] != meta:
                    if meta[0] != model.comfy_lora_meta_list[i][0]:
                        sd = load_state_dict_in_safetensors(meta[0])
                        model.comfy_lora_sd_list[i] = sd
                    model.comfy_lora_meta_list[i] = meta
                lora_to_be_composed.append(({k: v for k, v in model.comfy_lora_sd_list[i].items()}, meta[1]))

            composed_lora = compose_lora(lora_to_be_composed)

            if len(composed_lora) == 0:
                model.reset_lora()
            else:
                if "x_embedder.lora_A.weight" in composed_lora:
                    new_in_channels = composed_lora["x_embedder.lora_A.weight"].shape[1]
                    current_in_channels = model.x_embedder.in_features
                    if new_in_channels < current_in_channels:
                        model.reset_x_embedder()
                model.update_lora_params(composed_lora)

        if getattr(model, "_is_cached", False):
            if self._prev_timestep is None or self._prev_timestep < timestep_float:
                self._cache_context = create_cache_context()
            with cache_context(self._cache_context):
                out = model(
                    hidden_states=img,
                    encoder_hidden_states=context,
                    pooled_projections=y,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    controlnet_block_samples=None if control is None else control["input"],
                    controlnet_single_block_samples=None if control is None else control["output"],
                ).sample
        else:
            out = model(
                hidden_states=img,
                encoder_hidden_states=context,
                pooled_projections=y,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance if self.config["guidance_embed"] else None,
                controlnet_block_samples=None if control is None else control["input"],
                controlnet_single_block_samples=None if control is None else control["output"],
            ).sample

        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=patch_size, pw=patch_size)
        out = out[:, :, :h, :w]

        self._prev_timestep = timestep_float
        return out

    def load_state_dict(self, *args, **kwargs):
        return [], []


# ========== T5 ========== #


def _forward(self: "T5EncoderModel", input_ids: torch.LongTensor, *args, **kwargs):
    outputs = self.encoder(input_ids=input_ids, *args, **kwargs)
    return outputs.last_hidden_state


class WrappedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input: torch.Tensor, *args, **kwargs):
        return self.embedding(input)

    @property
    def weight(self):
        return self.embedding.weight


class SVDQT5(torch.nn.Module):
    """https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/v0.2.0/nodes/models/text_encoder.py#L45"""

    def __init__(self, path: str):
        super().__init__()

        transformer = NunchakuT5EncoderModel.from_pretrained(path)
        transformer.forward = types.MethodType(_forward, transformer)
        transformer.shared = WrappedEmbedding(transformer.shared)

        self.transformer = transformer
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
