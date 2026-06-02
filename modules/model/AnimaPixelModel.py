from contextlib import nullcontext
from random import Random

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from diffusers import AnimaTextConditioner, CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor


PROMPT_MAX_LENGTH = 512


class AnimaPixelDetailerHead(nn.Module):
    def __init__(self, in_channels: int = 3, feature_channels: int = 4096):
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


class AnimaPixelModel(BaseModel):
    tokenizer: Qwen2Tokenizer | None
    t5_tokenizer: T5TokenizerFast | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3Model | None
    text_conditioner: AnimaTextConditioner | None
    transformer: CosmosTransformer3DModel | None
    detailer_head: AnimaPixelDetailerHead | None

    text_encoder_autocast_context: torch.autocast | nullcontext
    text_encoder_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    lora_state_dict: dict | None

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)

        self.tokenizer = None
        self.t5_tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.text_conditioner = None
        self.transformer = None
        self.detailer_head = None

        self.text_encoder_autocast_context = nullcontext()
        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return []

    def vae_to(self, device: torch.device):
        pass

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)
            self.text_conditioner.to(device=device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

    def detailer_head_to(self, device: torch.device):
        self.detailer_head.to(device=device)

    def to(self, device: torch.device):
        self.text_encoder_to(device)
        self.transformer_to(device)
        self.detailer_head_to(device)

    def eval(self):
        if self.text_encoder is not None:
            self.text_encoder.eval()
            self.text_conditioner.eval()
        self.transformer.eval()
        self.detailer_head.eval()

    def create_pipeline(self):
        raise NotImplementedError("ANIMA_PIXEL uses a dedicated pixel sampler")

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            t5_tokens: Tensor = None,
            t5_tokens_mask: Tensor = None,
            text_encoder_layer_skip: int = 0,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> Tensor:
        del train_device, batch_size, rand, text_encoder_layer_skip

        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            tokenizer_output = self.tokenizer(
                text,
                max_length=PROMPT_MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

            t5_output = self.t5_tokenizer(
                text,
                max_length=PROMPT_MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            t5_tokens = t5_output.input_ids.to(self.text_encoder.device)
            t5_tokens_mask = t5_output.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is None and self.text_encoder is not None:
            if t5_tokens is None or t5_tokens_mask is None:
                raise ValueError("ANIMA_PIXEL text encoding requires T5 tokens when using cached Qwen tokens")

            with self.text_encoder_autocast_context:
                qwen_hidden = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask.float(),
                    output_hidden_states=False,
                ).last_hidden_state
                qwen_hidden = qwen_hidden * tokens_mask.to(qwen_hidden).unsqueeze(-1)
                text_encoder_output = self.text_conditioner(
                    source_hidden_states=qwen_hidden.to(dtype=self.text_conditioner.dtype),
                    target_input_ids=t5_tokens.to(self.text_encoder.device),
                    target_attention_mask=t5_tokens_mask.to(self.text_encoder.device),
                    source_attention_mask=tokens_mask,
                )

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError

        return text_encoder_output

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

        handle = self.transformer.norm_out.register_forward_pre_hook(capture_features)
        try:
            projected_flow = self.transformer(
                hidden_states=noisy_image.unsqueeze(2).to(dtype=self.train_dtype.torch_dtype()),
                timestep=timestep.to(dtype=torch.float32),
                encoder_hidden_states=text_encoder_output.to(dtype=self.train_dtype.torch_dtype()),
                return_dict=False,
            )[0][:, :, 0]
        finally:
            handle.remove()

        tokens = features.get("tokens")
        if tokens is None:
            raise RuntimeError("Failed to capture Anima pixel transformer features")

        batch_size, _, height, width = noisy_image.shape
        patch_size = int(self.transformer.config.patch_size[-1])
        feature_map = tokens.view(batch_size, 1, height // patch_size, width // patch_size, -1)
        feature_map = feature_map.squeeze(1).permute(0, 3, 1, 2).contiguous()
        return projected_flow + self.detailer_head(noisy_image, feature_map)
