from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.AnimaModel import AnimaModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_anima_transformer,
    enable_checkpointing_for_qwen3_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseAnimaSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta,
):
    LAYER_PRESETS = {
        "attn-mlp": {
            "patterns": [r"^(?!.*llm_adapter).*(attn|mlp).*"],
            "adapter_patterns": [r"^llm_adapter.*(attn|mlp).*"],
            "regex": True,
        },
        "attn-only": {
            "patterns": [r"^(?!.*llm_adapter).*attn.*"],
            "adapter_patterns": [r"^llm_adapter.*attn.*"],
            "regex": True,
        },
        "blocks": {
            "patterns": ["transformer_blocks"],
            "adapter_patterns": ["llm_adapter"],
            "regex": False,
        },
        "full": {
            "patterns": [r"^(?!.*llm_adapter).*"],
            "adapter_patterns": [r"^llm_adapter.*"],
            "regex": True,
        },
    }

    def setup_optimizations(
            self,
            model: AnimaModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = enable_checkpointing_for_anima_transformer(model.transformer, config)
            if model.text_encoder is not None:
                model.text_encoder_offload_conductor = enable_checkpointing_for_qwen3_encoder_layers(model.text_encoder, config)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().transformer,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
        ], config.enable_autocast_cache)

        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            [
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            ],
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)

    def predict(
            self,
            model: AnimaModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
            timestep: Tensor | None = None,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch["latent_image"].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                t5_tokens=batch.get("tokens_2"),
                t5_tokens_mask=batch.get("tokens_mask_2"),
                text_encoder_output=batch["text_encoder_hidden_state"]
                if "text_encoder_hidden_state" in batch and not config.train_text_encoder_or_embedding() else None,
                text_encoder_dropout_probability=config.text_encoder.dropout_probability,
            )

            scaled_latent_image = model.scale_latents(batch["latent_image"])
            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            noise_scheduler_config = model.noise_scheduler.config
            if isinstance(noise_scheduler_config, dict):
                num_train_timesteps = int(noise_scheduler_config["num_train_timesteps"])
            else:
                num_train_timesteps = int(noise_scheduler_config.num_train_timesteps)

            if timestep is None:
                timestep = self._get_timestep_discrete(
                    num_train_timesteps,
                    deterministic,
                    generator,
                    scaled_latent_image.shape[0],
                    config,
                    shift=config.timestep_shift,
                )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            predicted_flow = model.transformer(
                hidden_states=scaled_noisy_latent_image.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep.to(dtype=torch.float32) / num_train_timesteps,
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                return_dict=True,
            ).sample

            flow = latent_noise - scaled_latent_image
            model_output_data = {
                "loss_type": "target",
                "timestep": timestep,
                "predicted": predicted_flow,
                "target": flow,
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_tokens("7-prompt", batch["tokens"], model.qwen_tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: AnimaModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

    def prepare_text_caching(self, model: AnimaModel, config: TrainConfig):
        model.to(self.temp_device)
        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)

        model.eval()
        torch_gc()
