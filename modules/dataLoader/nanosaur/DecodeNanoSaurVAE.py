from mgds.PipelineModule import PipelineModule, RandomAccessPipelineModule

import torch


class DecodeNanoSaurVAE(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, in_name: str, out_name: str, model, autocast_contexts=None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.model = model
        self.autocast_contexts = [model.autocast_context] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        latent = self._get_previous_item(variation, self.in_name, index)
        if self.dtype is not None:
            latent = latent.to(dtype=self.dtype)
        with torch.no_grad(), self._all_contexts(self.autocast_contexts):
            image = self.model.decode_vae(latent.unsqueeze(0)).squeeze(0).clamp(-1, 1)
        return {self.out_name: image.detach().cpu()}
