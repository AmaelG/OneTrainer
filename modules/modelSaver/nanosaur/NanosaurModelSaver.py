import os
from pathlib import Path

from modules.model.NanosaurModel import NanosaurModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch
from safetensors.torch import save_file


class NanosaurModelSaver(DtypeModelSaverMixin):
    def _transformer_state_dict(self, model: NanosaurModel, dtype: torch.dtype | None) -> dict[str, torch.Tensor]:
        state_dict = model.transformer.state_dict()
        state_dict = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
        return self._convert_state_dict_dtype(state_dict, dtype)

    def _text_encoder_state_dict(self, model: NanosaurModel, dtype: torch.dtype | None) -> dict[str, torch.Tensor]:
        state_dict = {
            key: value.detach().cpu().contiguous()
            for key, value in model.text_encoder.state_dict().items()
            if key != "lm_head.weight"
        }
        spiece_model = getattr(model.tokenizer, "spiece_model", None)
        state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        if spiece_model is not None:
            state_dict["spiece_model"] = spiece_model.detach().cpu().contiguous()
        return state_dict

    def _vae_state_dict(self, model: NanosaurModel, dtype: torch.dtype | None) -> dict[str, torch.Tensor]:
        state_dict = {k: v.detach().cpu().contiguous() for k, v in model.vae.state_dict().items()}
        return self._convert_state_dict_dtype(state_dict, dtype)

    def _save_directory(self, model: NanosaurModel, destination: str, dtype: torch.dtype | None):
        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_file(self._transformer_state_dict(model, dtype), os.path.join(destination, "transformer.safetensor"))
        save_file(self._text_encoder_state_dict(model, dtype), os.path.join(destination, "text_encoder.safetensor"))
        save_file(self._vae_state_dict(model, dtype), os.path.join(destination, "vae.safetensor"))

    def save(self, model: NanosaurModel, output_model_format: ModelFormat, output_model_destination: str, dtype: torch.dtype | None):
        match output_model_format:
            case ModelFormat.INTERNAL:
                self._save_directory(model, output_model_destination, None)
            case ModelFormat.SAFETENSORS:
                self._save_directory(model, output_model_destination, dtype)
            case _:
                raise NotImplementedError(f"Nanosaur does not support {output_model_format} output")
