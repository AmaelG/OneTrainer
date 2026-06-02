import torch

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class NormalizeImageRange(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, image_name: str):
        super().__init__()
        self.image_name = image_name

    def length(self) -> int:
        return self._get_previous_length(self.image_name)

    def get_inputs(self) -> list[str]:
        return [self.image_name]

    def get_outputs(self) -> list[str]:
        return [self.image_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.image_name, index)
        if image is not None and torch.is_tensor(image) and image.numel() > 0:
            image_max = image.detach().amax()
            if image_max > 1.01:
                # MGDS historically divided all integer images by 255. 16-bit PNGs can arrive as 0..257.
                image = (image / 257.0).clamp(0.0, 1.0)

        return {self.image_name: image}
