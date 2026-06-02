from abc import ABCMeta
from random import Random

import torch
from torch import Tensor

import modules.util.multi_gpu_util as multi
from modules.model.AnimaPixelModel import AnimaPixelModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import enable_checkpointing_for_qwen3_encoder_layers, enable_checkpointing_for_qwen_transformer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.ModuleFilter import ModuleFilter
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress


class BaseAnimaPixelSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta,
):
    LAYER_PRESETS = {
        "l2p-shallow": {
            "patterns": [
                r"^transformer\.patch_embed\.",
                r"^transformer\.transformer_blocks\.(0|1|2|3|4|23|24|25|26|27)\.",
                r"^detailer_head\.",
            ],
            "adapter_patterns": [],
            "regex": True,
        },
        "l2p-n3-targeted": {
            "patterns": [
                r"^transformer\.patch_embed\.",
                r"^transformer\.norm_out\.",
                r"^transformer\.proj_out\.",
                r"^detailer_head\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.norm1\.(linear_1|linear_2)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.attn1\.(norm_q|norm_k|to_q|to_k|to_v|to_out\.0)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.norm2\.(linear_1|linear_2)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.attn2\.(norm_q|to_q|to_out\.0)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.norm3\.(linear_1|linear_2)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.ff\.net\.(0\.proj|2)\.",
            ],
            "adapter_patterns": [],
            "regex": True,
        },
        "l2p-n3-no-norm": {
            "patterns": [
                r"^transformer\.patch_embed\.",
                r"^transformer\.proj_out\.",
                r"^detailer_head\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.norm1\.(linear_1|linear_2)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.attn1\.(to_q|to_k|to_v|to_out\.0)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.norm2\.(linear_1|linear_2)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.attn2\.(to_q|to_out\.0)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.norm3\.(linear_1|linear_2)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.ff\.net\.(0\.proj|2)\.",
            ],
            "adapter_patterns": [],
            "regex": True,
        },
        "l2p-n3-no-norm-no-adaln": {
            "patterns": [
                r"^transformer\.patch_embed\.",
                r"^transformer\.proj_out\.",
                r"^detailer_head\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.attn1\.(to_q|to_k|to_v|to_out\.0)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.attn2\.(to_q|to_out\.0)\.",
                r"^transformer\.transformer_blocks\.(0|1|2|25|26|27)\.ff\.net\.(0\.proj|2)\.",
            ],
            "adapter_patterns": [],
            "regex": True,
        },
    }

    QUANTIZATION_LAYER_PRESETS = {
        "l2p-frozen-middle": {
            "patterns": [
                r"^transformer\.transformer_blocks\.(5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22)\.",
            ],
            "adapter_patterns": [],
            "regex": True,
        },
    }

    @staticmethod
    def default_train_filters() -> list[ModuleFilter]:
        preset = BaseAnimaPixelSetup.LAYER_PRESETS["l2p-n3-targeted"]
        return [
            ModuleFilter(pattern, use_regex=bool(preset.get("regex", False)))
            for pattern in preset.get("patterns", [])
        ]

    @staticmethod
    def train_filters(config: TrainConfig) -> list[ModuleFilter]:
        preset = BaseAnimaPixelSetup.LAYER_PRESETS.get(config.layer_filter_preset)
        if isinstance(preset, dict):
            return [
                ModuleFilter(pattern, use_regex=bool(preset.get("regex", False)))
                for pattern in preset.get("patterns", [])
            ]
        if isinstance(preset, list):
            return [ModuleFilter(pattern, use_regex=False) for pattern in preset]
        if config.layer_filter.strip():
            return ModuleFilter.create(config)
        return BaseAnimaPixelSetup.default_train_filters()

    @staticmethod
    def matches_filter(name: str, prefix: str, filters: list[ModuleFilter]) -> bool:
        name = name.replace(".checkpoint.", ".")
        return any(f.matches(name) or f.matches(f"{prefix}.{name}") for f in filters)

    def setup_optimizations(self, model: AnimaPixelModel, config: TrainConfig):
        if config.compile:
            # Anima Pixel mixes frozen/trainable Cosmos blocks and validation/sample passes.
            # Allow the shared block forward to keep a few expected compiled variants.
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)

        if config.transformer.checkpointing_or_offloading_enabled():
            model.transformer_offload_conductor = enable_checkpointing_for_qwen_transformer(
                model.transformer, config, config.transformer,
            )
        if model.text_encoder is not None and config.text_encoder.checkpointing_or_offloading_enabled():
            model.text_encoder_offload_conductor = enable_checkpointing_for_qwen3_encoder_layers(
                model.text_encoder, config, config.text_encoder,
            )

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device, config.train_dtype, config.enable_autocast_cache,
        )
        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)

    def predict(
            self,
            model: AnimaPixelModel,
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

            pixel_image = batch["pixel_image"].to(device=self.train_device, dtype=model.train_dtype.torch_dtype())
            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=pixel_image.shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                t5_tokens=batch.get("t5_tokens"),
                t5_tokens_mask=batch.get("t5_tokens_mask"),
                text_encoder_output=batch["text_encoder_hidden_state"]
                if "text_encoder_hidden_state" in batch and not config.train_text_encoder_or_embedding() else None,
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
            )

            noise = self._create_noise(pixel_image, config, generator)
            num_train_timesteps = int(model.noise_scheduler.config["num_train_timesteps"])
            if model.noise_scheduler.timesteps.shape[-1] != num_train_timesteps:
                model.noise_scheduler.set_timesteps(num_train_timesteps, device=self.train_device)

            if timestep is None:
                timestep = self._get_timestep_discrete(
                    num_train_timesteps,
                    deterministic,
                    generator,
                    pixel_image.shape[0],
                    config,
                    shift=config.timestep_shift,
                )

            noisy_image, _sigma = self._add_noise_discrete(
                pixel_image,
                noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            predicted_flow = model.predict_pixel_flow(
                noisy_image=noisy_image,
                timestep=timestep.to(dtype=torch.float32) / num_train_timesteps,
                text_encoder_output=text_encoder_output,
            )
            flow = noise - pixel_image
            return {
                "loss_type": "target",
                "timestep": timestep,
                "pixel_image": pixel_image,
                "noisy_image": noisy_image,
                "predicted": predicted_flow,
                "target": flow,
            }

    def calculate_sample_losses(self, model: AnimaPixelModel, batch: dict, data: dict, config: TrainConfig) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        )

    def calculate_loss(self, model: AnimaPixelModel, batch: dict, data: dict, config: TrainConfig) -> Tensor:
        return self.calculate_sample_losses(model, batch, data, config).mean()

    def prepare_text_caching(self, model: AnimaPixelModel, config: TrainConfig):
        model.to(self.temp_device)
        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)
        model.eval()
        torch_gc()
