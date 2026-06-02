import os
import traceback

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from diffusers import (
    AnimaTextConditioner,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast

from modules.model.AnimaPixelModel import AnimaPixelDetailerHead, AnimaPixelModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class AnimaPixelModelLoader(HFModelLoaderMixin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def __convert_pos_embed(source_value: torch.Tensor, target_value: torch.Tensor, key: str) -> torch.Tensor | None:
        if not key.startswith("learnable_pos_embed."):
            return None
        if source_value.ndim != 2 or target_value.ndim != 2 or source_value.shape[1] != target_value.shape[1]:
            return None

        if source_value.shape[0] == target_value.shape[0]:
            return source_value

        if target_value.shape[0] == 1:
            return source_value.mean(dim=0, keepdim=True)

        return F.interpolate(
            source_value.transpose(0, 1).unsqueeze(0),
            size=target_value.shape[0],
            mode="linear",
            align_corners=True,
        ).squeeze(0).transpose(0, 1).contiguous()

    @staticmethod
    def __pixel_transformer_from_base(base_transformer: CosmosTransformer3DModel) -> CosmosTransformer3DModel:
        config = base_transformer.config
        pixel_transformer = CosmosTransformer3DModel(
            in_channels=3,
            out_channels=3,
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            num_layers=config.num_layers,
            mlp_ratio=config.mlp_ratio,
            text_embed_dim=config.text_embed_dim,
            adaln_lora_dim=config.adaln_lora_dim,
            max_size=(1, 2048, 2048),
            patch_size=(1, 16, 16),
            rope_scale=tuple(config.rope_scale),
            concat_padding_mask=False,
            extra_pos_embed_type=config.extra_pos_embed_type,
            use_crossattn_projection=config.use_crossattn_projection,
            crossattn_proj_in_channels=config.crossattn_proj_in_channels,
            encoder_hidden_states_channels=config.encoder_hidden_states_channels,
            controlnet_block_every_n=config.controlnet_block_every_n,
            img_context_dim_in=config.img_context_dim_in,
            img_context_num_tokens=config.img_context_num_tokens,
            img_context_dim_out=config.img_context_dim_out,
        )

        source_state = base_transformer.state_dict()
        target_state = pixel_transformer.state_dict()
        converted_state = {}
        copied = 0
        initialized = 0
        for key, target_value in target_state.items():
            source_value = source_state.get(key)
            if source_value is not None and tuple(source_value.shape) == tuple(target_value.shape):
                converted_state[key] = source_value
                copied += 1
            elif source_value is not None and (converted_pos_embed := AnimaPixelModelLoader.__convert_pos_embed(source_value, target_value, key)) is not None:
                converted_state[key] = converted_pos_embed
                copied += 1
            else:
                converted_state[key] = target_value
                initialized += 1

        pixel_transformer.load_state_dict(converted_state, strict=True)
        print(f"ANIMA_PIXEL copied {copied} base tensors and initialized {initialized} pixel tensors")
        return pixel_transformer

    @staticmethod
    def __load_overlay(model: AnimaPixelModel, overlay_path: str):
        if not overlay_path:
            return
        if os.path.isdir(overlay_path):
            overlay_path = os.path.join(overlay_path, "anima_pixel_delta.safetensors")
        if not os.path.isfile(overlay_path):
            raise FileNotFoundError(f"ANIMA_PIXEL overlay checkpoint not found: {overlay_path}")

        state_dict = load_file(overlay_path, device="cpu")
        state_dict = {key.replace(".checkpoint.", "."): value for key, value in state_dict.items()}
        transformer_state = {
            key.removeprefix("transformer."): value
            for key, value in state_dict.items()
            if key.startswith("transformer.")
        }
        detailer_state = {
            key.removeprefix("detailer_head."): value
            for key, value in state_dict.items()
            if key.startswith("detailer_head.")
        }
        unknown_keys = [key for key in state_dict.keys() if not key.startswith(("transformer.", "detailer_head."))]
        if unknown_keys:
            raise RuntimeError(f"Unexpected ANIMA_PIXEL overlay keys: {unknown_keys}")

        if transformer_state:
            missing, unexpected = model.transformer.load_state_dict(transformer_state, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected ANIMA_PIXEL transformer overlay keys: {unexpected}")
            print(f"Loaded ANIMA_PIXEL transformer overlay tensors: {len(transformer_state)}; missing base keys: {len(missing)}")
        if detailer_state:
            missing, unexpected = model.detailer_head.load_state_dict(detailer_state, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected ANIMA_PIXEL detailer overlay keys: {unexpected}")
            print(f"Loaded ANIMA_PIXEL detailer overlay tensors: {len(detailer_state)}; missing base keys: {len(missing)}")

    def __load_diffusers(
            self,
            model: AnimaPixelModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        del vae_model_name

        tokenizer = Qwen2Tokenizer.from_pretrained(base_model_name, subfolder="tokenizer")
        t5_tokenizer = T5TokenizerFast.from_pretrained(base_model_name, subfolder="t5_tokenizer")
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_name, subfolder="scheduler")

        text_encoder = self._load_transformers_sub_module(
            Qwen3Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        text_conditioner = AnimaTextConditioner.from_pretrained(
            base_model_name,
            subfolder="text_conditioner",
            torch_dtype=torch.bfloat16,
        )

        base_transformer = self._load_diffusers_sub_module(
            CosmosTransformer3DModel,
            weight_dtypes.train_dtype,
            weight_dtypes.train_dtype,
            base_model_name,
            "transformer",
            QuantizationConfig.default_values(),
        )
        transformer = self.__pixel_transformer_from_base(base_transformer)
        transformer = self._convert_diffusers_sub_module_to_dtype(
            transformer,
            weight_dtypes.transformer,
            weight_dtypes.train_dtype,
            quantization,
        )
        detailer_head = AnimaPixelDetailerHead(
            feature_channels=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
        )
        detailer_head.to(dtype=weight_dtypes.train_dtype.torch_dtype())

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.t5_tokenizer = t5_tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.text_conditioner = text_conditioner
        model.transformer = transformer
        model.detailer_head = detailer_head

        self.__load_overlay(model, transformer_model_name)

    def load(
            self,
            model: AnimaPixelModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []
        try:
            self.__load_diffusers(
                model,
                model_type,
                weight_dtypes,
                model_names.base_model,
                model_names.transformer_model,
                model_names.vae_model,
                quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load ANIMA_PIXEL model: " + model_names.base_model)
