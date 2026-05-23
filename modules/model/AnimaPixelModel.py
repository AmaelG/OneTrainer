from __future__ import annotations

from contextlib import nullcontext
from random import Random

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from diffusers import AnimaTransformer3DModel, FlowMatchEulerDiscreteScheduler
from transformers import PreTrainedModel, PreTrainedTokenizer

from modules.model.AnimaModel import AnimaModel
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType


class AnimaPixelDetailerHead(nn.Module):
    def __init__(self, in_channels: int = 3, feature_channels: int = 2048):
        super().__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.SiLU())
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.SiLU())
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.SiLU())
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.SiLU())
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.bottleneck = nn.Sequential(nn.Conv2d(512 + feature_channels, 512, kernel_size=1), nn.SiLU())

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.dec4 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.SiLU())
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.dec3 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.SiLU())
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.dec2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.SiLU())
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x: Tensor, features: Tensor) -> Tensor:
        enc1 = self.enc1(x)
        p1 = self.pool1(enc1)
        enc2 = self.enc2(p1)
        p2 = self.pool2(enc2)
        enc3 = self.enc3(p2)
        p3 = self.pool3(enc3)
        enc4 = self.enc4(p3)
        p4 = self.pool4(enc4)

        if features.shape[-2:] != p4.shape[-2:]:
            features = F.interpolate(features, size=p4.shape[-2:], mode="nearest")
        bottleneck = self.bottleneck(torch.cat([p4, features], dim=1))

        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        return self.out_conv(dec1)


class AnimaPixelModel(AnimaModel):
    qwen_tokenizer: PreTrainedTokenizer | None
    t5_tokenizer: PreTrainedTokenizer | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: PreTrainedModel | None
    transformer: AnimaTransformer3DModel | None
    detailer_head: AnimaPixelDetailerHead | None

    text_encoder_autocast_context: torch.autocast | nullcontext
    text_encoder_train_dtype: DataType

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)
        self.detailer_head = None

    def adapters(self):
        return []

    def vae_to(self, device: torch.device):
        if self.vae is not None:
            self.vae.to(device=device)

    def detailer_head_to(self, device: torch.device):
        if self.detailer_head is not None:
            self.detailer_head.to(device=device)

    def to(self, device: torch.device):
        self.text_encoder_to(device)
        self.transformer_to(device)
        self.detailer_head_to(device)

    def eval(self):
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.transformer is not None:
            self.transformer.eval()
        if self.detailer_head is not None:
            self.detailer_head.eval()

    def create_pipeline(self):
        raise NotImplementedError("ANIMA_PIXEL uses a custom pixel sampler, not the latent AnimaPipeline")

    def predict_pixel_flow(
            self,
            noisy_image: Tensor,
            timestep: Tensor,
            text_encoder_output: Tensor,
    ) -> Tensor:
        if self.transformer is None or self.detailer_head is None:
            raise RuntimeError("AnimaPixelModel is not fully initialized")

        features: dict[str, Tensor] = {}

        def capture_features(_module, inputs):
            features["tokens"] = inputs[0]

        handle = self.transformer.core.norm_out.register_forward_pre_hook(capture_features)
        try:
            _ = self.transformer(
                hidden_states=noisy_image.unsqueeze(2).to(dtype=self.train_dtype.torch_dtype()),
                timestep=timestep.to(dtype=torch.float32),
                encoder_hidden_states=text_encoder_output.to(dtype=self.train_dtype.torch_dtype()),
                return_dict=False,
            )[0]
        finally:
            handle.remove()

        tokens = features.get("tokens")
        if tokens is None:
            raise RuntimeError("Failed to capture Anima pixel transformer features")

        batch_size, _, height, width = noisy_image.shape
        patch_size = int(self.transformer.config.patch_size[-1])
        feature_map = tokens.view(batch_size, 1, height // patch_size, width // patch_size, -1)
        feature_map = feature_map.squeeze(1).permute(0, 3, 1, 2).contiguous()
        return self.detailer_head(noisy_image, feature_map)

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] | None = None,
            tokens: Tensor | None = None,
            tokens_mask: Tensor | None = None,
            t5_tokens: Tensor | None = None,
            t5_tokens_mask: Tensor | None = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor | None = None,
    ) -> Tensor:
        return super().encode_text(
            train_device=train_device,
            batch_size=batch_size,
            rand=rand,
            text=text,
            tokens=tokens,
            tokens_mask=tokens_mask,
            t5_tokens=t5_tokens,
            t5_tokens_mask=t5_tokens_mask,
            text_encoder_dropout_probability=text_encoder_dropout_probability,
            text_encoder_output=text_encoder_output,
        )
