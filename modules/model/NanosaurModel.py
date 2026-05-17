from contextlib import nullcontext

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType

import torch
from torch import Tensor, nn


from modules.model.nanosaur.model_lora import (
    LATENT_SCALE,
    LATENT_SHIFT,
    TEXT_MAX_LENGTH,
    LoraNanoSaurTransformer2DModel,
    NanoSaurSentencePieceTokenizer,
)
from modules.model.nanosaur.vae import NanoSaurVAE


class NanosaurModel(BaseModel):
    tokenizer: NanoSaurSentencePieceTokenizer | None
    text_encoder: nn.Module | None
    vae: NanoSaurVAE | None
    transformer: LoraNanoSaurTransformer2DModel | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None
    autocast_context: torch.autocast | nullcontext

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)

        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self.transformer_lora = None
        self.lora_state_dict = None
        self.autocast_context = nullcontext()
        self.train_dtype = DataType.FLOAT_32

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [self.transformer_lora] if a is not None]

    def vae_to(self, device: torch.device):
        if self.vae is not None:
            self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            self.text_encoder.to(device=device)

    def transformer_to(self, device: torch.device):
        if self.transformer is not None:
            self.transformer.to(device=device)
        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        if self.vae is not None:
            self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.transformer is not None:
            self.transformer.eval()

    def encode_vae(self, image: Tensor) -> Tensor:
        dtype = self.train_dtype.torch_dtype() or image.dtype
        image = image.to(device=next(self.vae.parameters()).device, dtype=dtype)
        return (self.vae.encode(image) + LATENT_SHIFT) / LATENT_SCALE

    def decode_vae(self, latent: Tensor) -> Tensor:
        latent = (latent * LATENT_SCALE) - LATENT_SHIFT
        dtype = self.train_dtype.torch_dtype() or latent.dtype
        latent = latent.to(device=next(self.vae.parameters()).device, dtype=dtype)
        return self.vae.decode(latent)

    @torch.no_grad()
    def encode_text(
            self,
            text: str | list[str],
            train_device: torch.device,
    ) -> Tensor:
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError("Nanosaur tokenizer/text encoder is not initialized")

        captions = [text] if isinstance(text, str) else list(text)
        tokens = self.tokenizer(captions, device=train_device)
        outputs = self.text_encoder(**tokens, output_hidden_states=True)
        return outputs.hidden_states[-1]

    @staticmethod
    def text_max_length() -> int:
        return TEXT_MAX_LENGTH
