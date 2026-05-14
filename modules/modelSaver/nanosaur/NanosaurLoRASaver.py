import os
from pathlib import Path

from modules.model.NanosaurModel import NanosaurModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch
from safetensors.torch import save_file


class NanosaurLoRASaver(DtypeModelSaverMixin):
    def _get_state_dict(self, model: NanosaurModel) -> dict[str, torch.Tensor]:
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict
        return state_dict

    def _to_comfy_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        comfy_state_dict = {}
        prefix = "lora_transformer."
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            comfy_key = "diffusion_model." + key.removeprefix(prefix)
            comfy_state_dict[comfy_key] = value.detach().cpu().contiguous()
        return comfy_state_dict

    def save(self, model: NanosaurModel, output_model_format: ModelFormat, output_model_destination: str, dtype: torch.dtype | None):
        state_dict = self._convert_state_dict_dtype(self._get_state_dict(model), dtype)

        if output_model_format == ModelFormat.INTERNAL:
            destination = os.path.join(output_model_destination, "lora", "lora.safetensors")
            save_state_dict = state_dict
        else:
            destination = output_model_destination
            save_state_dict = self._to_comfy_state_dict(state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))
