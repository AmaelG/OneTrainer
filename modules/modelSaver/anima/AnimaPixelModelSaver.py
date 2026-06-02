import os
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

from modules.model.AnimaPixelModel import AnimaPixelModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat


class AnimaPixelModelSaver(DtypeModelSaverMixin):
    @staticmethod
    def __format_block_ranges(blocks: set[int]) -> str:
        if not blocks:
            return ""

        ranges = []
        start = None
        previous = None
        for block in sorted(blocks):
            if start is None:
                start = block
            elif previous is not None and block != previous + 1:
                ranges.append(str(start) if start == previous else f"{start}-{previous}")
                start = block
            previous = block

        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        return ",".join(ranges)

    @staticmethod
    def __normalize_key(name: str) -> str:
        return name.replace(".checkpoint.", ".")

    @staticmethod
    def __trainable_blocks(state_dict: dict[str, torch.Tensor]) -> str:
        blocks = set()
        for key in state_dict.keys():
            match = re.match(r"^transformer\.transformer_blocks\.(\d+)\.", key)
            if match:
                blocks.add(int(match.group(1)))
        return AnimaPixelModelSaver.__format_block_ranges(blocks)

    def __trainable_state_dict(self, model: AnimaPixelModel, dtype: torch.dtype | None) -> dict[str, torch.Tensor]:
        state_dict = {}
        for name, param in model.transformer.named_parameters():
            if param.requires_grad:
                state_dict[f"transformer.{self.__normalize_key(name)}"] = param.detach().cpu().contiguous()
        for name, param in model.detailer_head.named_parameters():
            if param.requires_grad:
                state_dict[f"detailer_head.{self.__normalize_key(name)}"] = param.detach().cpu().contiguous()

        state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(state_dict)
        return state_dict

    def save(self, model: AnimaPixelModel, output_model_format: ModelFormat, output_model_destination: str, dtype: torch.dtype | None):
        if output_model_format not in (ModelFormat.SAFETENSORS, ModelFormat.INTERNAL):
            raise NotImplementedError("ANIMA_PIXEL currently only supports safetensors subset-overlay export")

        state_dict = self.__trainable_state_dict(model, dtype)
        destination = output_model_destination
        if output_model_format == ModelFormat.INTERNAL:
            os.makedirs(Path(output_model_destination).absolute(), exist_ok=True)
            destination = os.path.join(output_model_destination, "anima_pixel_delta.safetensors")

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        metadata = self._create_safetensors_header(model, state_dict)
        metadata.update({
            "modelspec.architecture": "AnimaPixel/delta",
            "onetrainer.checkpoint_type": "subset_overlay",
            "onetrainer.base_model_type": "ANIMA",
            "onetrainer.pixel_patch_size": "16",
            "onetrainer.trainable_blocks": self.__trainable_blocks(state_dict),
        })
        save_file(state_dict, destination, metadata)
