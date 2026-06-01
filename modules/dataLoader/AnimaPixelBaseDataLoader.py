import os

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.EncodeQwenText import EncodeQwenText
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.Tokenize import Tokenize

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.NormalizeImageRange import NormalizeImageRange
from modules.model.AnimaPixelModel import AnimaPixelModel
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseAnimaPixelSetup import BaseAnimaPixelSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress


class AnimaPixelBaseDataLoader(BaseDataLoader, DataLoaderText2ImageMixin):
    def _preparation_modules(self, config: TrainConfig, model: AnimaPixelModel):
        prompt_max_length = model.prompt_max_length()
        rescale_image = RescaleImageChannels(
            image_in_name="image", image_out_name="pixel_image", in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1
        )
        tokenize_prompt = Tokenize(
            in_name="prompt",
            tokens_out_name="tokens",
            mask_out_name="tokens_mask",
            tokenizer=model.qwen_tokenizer,
            max_token_length=prompt_max_length,
        )
        tokenize_prompt_2 = Tokenize(
            in_name="prompt",
            tokens_out_name="tokens_2",
            mask_out_name="tokens_mask_2",
            tokenizer=model.t5_tokenizer,
            max_token_length=prompt_max_length,
        )
        encode_prompt = EncodeQwenText(
            tokens_name="tokens",
            tokens_attention_mask_in_name="tokens_mask",
            hidden_state_out_name="text_encoder_hidden_state",
            tokens_attention_mask_out_name="tokens_mask",
            text_encoder=model.text_encoder,
            hidden_state_output_index=-1,
            autocast_contexts=[model.text_encoder_autocast_context],
            dtype=model.text_encoder_train_dtype.torch_dtype(),
        )

        modules = [rescale_image, tokenize_prompt, tokenize_prompt_2]
        if not config.train_text_encoder_or_embedding():
            modules.append(encode_prompt)
        return modules

    def _cache_modules(self, config: TrainConfig, model: AnimaPixelModel, model_setup: BaseAnimaPixelSetup):
        image_split_names = ["pixel_image", "original_resolution", "crop_offset"]
        image_aggregate_names = ["crop_resolution", "image_path"]
        text_split_names = []
        sort_names = image_aggregate_names + image_split_names + [
            "prompt", "tokens", "tokens_mask", "tokens_2", "tokens_mask_2", "text_encoder_hidden_state", "concept"
        ]
        if not config.train_text_encoder_or_embedding():
            text_split_names += ["tokens", "tokens_mask", "tokens_2", "tokens_mask_2", "text_encoder_hidden_state"]

        return self._cache_modules_from_names(
            model,
            model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=not config.train_text_encoder_or_embedding(),
            before_cache_image_fun=lambda: None,
        )

    def _output_modules(self, config: TrainConfig, model: AnimaPixelModel, model_setup: BaseAnimaPixelSetup):
        output_names = [
            "image_path",
            "pixel_image",
            "prompt",
            "tokens",
            "tokens_mask",
            "tokens_2",
            "tokens_mask_2",
            "original_resolution",
            "crop_resolution",
            "crop_offset",
        ]
        if not config.train_text_encoder_or_embedding():
            output_names.append("text_encoder_hidden_state")

        return self._output_modules_from_out_names(
            model,
            model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            before_cache_image_fun=lambda: None,
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: AnimaPixelModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")
        debug_image = RescaleImageChannels(
            image_in_name="pixel_image",
            image_out_name="debug_pixel_image",
            in_range_min=-1,
            in_range_max=1,
            out_range_min=-1,
            out_range_max=1,
        )
        save_image = SaveImage(
            image_in_name="debug_pixel_image",
            original_path_in_name="image_path",
            path=debug_dir,
            in_range_min=-1,
            in_range_max=1,
        )
        decode_prompt = DecodeTokens(in_name="tokens", out_name="decoded_prompt", tokenizer=model.qwen_tokenizer)
        save_prompt = SaveText(text_in_name="decoded_prompt", original_path_in_name="image_path", path=debug_dir)
        return [debug_image, save_image, decode_prompt, save_prompt]

    def _create_dataset(
            self,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        enumerate_input = self._enumerate_input_modules(config, allow_videos=False)
        load_input = self._load_input_modules(config, model.train_dtype, vae_frame_dim=False)
        load_input.append(NormalizeImageRange("image"))
        mask_augmentation = self._mask_augmentation_modules(config)
        aspect_bucketing_in = self._aspect_bucketing_in(config, 16, False)
        crop_modules = self._crop_modules(config)
        augmentation_modules = self._augmentation_modules(config)
        inpainting_modules = self._inpainting_modules(config)
        preparation_modules = self._preparation_modules(config, model)
        cache_modules = self._cache_modules(config, model, model_setup)
        debug_modules = self._debug_modules(config, model)
        output_modules = self._output_modules(config, model, model_setup)

        return self._create_mgds(
            config,
            [
                enumerate_input,
                load_input,
                mask_augmentation,
                aspect_bucketing_in,
                crop_modules,
                augmentation_modules,
                inpainting_modules,
                preparation_modules,
                cache_modules,
                debug_modules if config.debug_mode else None,
                output_modules,
            ],
            train_progress,
            is_validation,
        )


factory.register(BaseDataLoader, AnimaPixelBaseDataLoader, ModelType.ANIMA_PIXEL)
