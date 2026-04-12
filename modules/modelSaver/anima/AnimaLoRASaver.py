import os
import re
from pathlib import Path

from modules.model.BaseModel import BaseModel
from modules.model.AnimaModel import AnimaModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor

from safetensors.torch import save_file


class AnimaLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list | None:
        del model
        return None

    def _get_state_dict(
            self,
            model: BaseModel,
    ) -> dict[str, Tensor]:
        assert isinstance(model, AnimaModel)
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict
        return state_dict

    @staticmethod
    def __to_comfy_key(key: str) -> str | None:
        match = re.match(
            r"^transformer\.(?:core\.)?transformer_blocks\.(\d+)\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)(\..+)$",
            key,
        )
        if match is not None:
            block_index, attn_block, projection, suffix = match.groups()

            attn_name = "self_attn" if attn_block == "attn1" else "cross_attn"
            projection_name = {
                "to_q": "q_proj",
                "to_k": "k_proj",
                "to_v": "v_proj",
                "to_out.0": "output_proj",
            }[projection]

            return f"lora_unet_blocks_{block_index}_{attn_name}_{projection_name}{suffix}"

        match = re.match(
            r"^transformer\.llm_adapter\.blocks\.(\d+)\.(self_attn|cross_attn)\.(q_proj|k_proj|v_proj|o_proj)(\..+)$",
            key,
        )
        if match is not None:
            block_index, attn_block, projection, suffix = match.groups()
            return f"lora_unet_llm_adapter_blocks_{block_index}_{attn_block}_{projection}{suffix}"

        match = re.match(
            r"^transformer\.llm_adapter\.blocks\.(\d+)\.mlp\.(0|2)(\..+)$",
            key,
        )
        if match is not None:
            block_index, mlp_layer, suffix = match.groups()
            return f"lora_unet_llm_adapter_blocks_{block_index}_mlp_{mlp_layer}{suffix}"

        return None

    def __save_comfy_safetensors(
            self,
            model: AnimaModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = self._get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)

        comfy_state_dict = {}
        for key, value in save_state_dict.items():
            comfy_key = self.__to_comfy_key(key)
            if comfy_key is not None:
                comfy_state_dict[comfy_key] = value

        if not comfy_state_dict:
            raise RuntimeError("No Anima transformer LoRA keys found for Comfy export")

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(comfy_state_dict, destination, self._create_safetensors_header(model, comfy_state_dict))

    def save(
            self,
            model: AnimaModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        if output_model_format == ModelFormat.COMFY_LORA:
            self.__save_comfy_safetensors(model, output_model_destination, dtype)
        else:
            self._save(model, output_model_format, output_model_destination, dtype, enable_omi_format=True)
