import math

from modules.model.NanosaurModel import NanosaurModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseNanosaurSetup(BaseModelSetup, ModelSetupDiffusionLossMixin, ModelSetupText2ImageMixin):
    LAYER_PRESETS = {
        "creator-default": ["blocks.", "text_refine_blocks.", "dec_net.res_blocks."],
        "transformer-full": [],
    }

    def setup_optimizations(self, model: NanosaurModel, config: TrainConfig):
        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().transformer,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
        ], config.enable_autocast_cache)

    def _sample_timesteps(self, batch_size: int, device: torch.device, dtype: torch.dtype, deterministic: bool) -> Tensor:
        if deterministic:
            return torch.full((batch_size,), 0.5, device=device, dtype=dtype)
        mu = math.log(2.0)
        return torch.sigmoid(torch.randn((batch_size,), device=device, dtype=dtype) + mu)

    def predict(
            self,
            model: NanosaurModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
            timestep: Tensor | None = None,
    ) -> dict:
        del train_progress
        with model.autocast_context:
            train_dtype = model.train_dtype.torch_dtype() or torch.float32
            latents = batch["latent_image"].to(device=self.train_device, dtype=train_dtype)

            if "text_embedding" in batch and not config.train_text_encoder_or_embedding():
                cond = batch["text_embedding"].to(device=self.train_device, dtype=train_dtype)
            else:
                prompt = batch["prompt"]
                cond = model.encode_text(prompt, self.train_device).to(dtype=train_dtype)

            if not deterministic and config.text_encoder.dropout_probability > 0:
                uncond = model.encode_text([""], self.train_device).to(device=self.train_device, dtype=cond.dtype)
                mask = torch.rand(cond.size(0), device=self.train_device) < config.text_encoder.dropout_probability
                if mask.any():
                    cond = cond.clone()
                    cond[mask] = uncond.expand(cond.size(0), -1, -1)[mask]

            if timestep is None:
                t = self._sample_timesteps(latents.size(0), self.train_device, latents.dtype, deterministic)
            else:
                # OneTrainer validation timesteps are specified in the usual 0..1000 range.
                # Nanosaur trains on continuous rectified-flow time in 0..1.
                t = (timestep.to(device=self.train_device, dtype=latents.dtype) / 1000.0).clamp(0.0, 1.0)

            shape = [latents.size(0)] + [1] * (latents.dim() - 1)
            noise = torch.randn_like(latents)
            noisy_latents = (1 - t.view(shape)) * latents + t.view(shape) * noise
            predicted_x0, _ = model.transformer(noisy_latents, t, cond, return_x0=True)
            t_clamped = (t + 0.05).view(shape)
            predicted_velocity = (noisy_latents - predicted_x0) / t_clamped
            target_velocity = (noisy_latents - latents) / t_clamped

            return {
                "loss_type": "target",
                "predicted": predicted_velocity,
                "target": target_velocity,
            }

    def calculate_loss(self, model: NanosaurModel, batch: dict, data: dict, config: TrainConfig) -> Tensor:
        return self._diffusion_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
        ).mean()

    def prepare_text_caching(self, model: NanosaurModel, config: TrainConfig):
        model.to(self.temp_device)
        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)
        model.eval()
        torch_gc()
