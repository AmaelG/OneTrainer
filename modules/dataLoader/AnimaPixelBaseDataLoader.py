import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.NormalizeImageRange import NormalizeImageRange
from modules.model.AnimaPixelModel import PROMPT_MAX_LENGTH, AnimaPixelModel
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseAnimaPixelSetup import BaseAnimaPixelSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.EncodeAnimaText import EncodeAnimaText
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.Tokenize import Tokenize


class AnimaPixelBaseDataLoader(BaseDataLoader, DataLoaderText2ImageMixin):
    def _preparation_modules(self, config: TrainConfig, model: AnimaPixelModel):
        rescale_image = RescaleImageChannels(
            image_in_name="image",
            image_out_name="pixel_image",
            in_range_min=0,
            in_range_max=1,
            out_range_min=-1,
            out_range_max=1,
        )
        tokenize_prompt = Tokenize(
            in_name="prompt",
            tokens_out_name="tokens",
            mask_out_name="tokens_mask",
            tokenizer=model.tokenizer,
            max_token_length=PROMPT_MAX_LENGTH,
        )
        tokenize_t5 = Tokenize(
            in_name="prompt",
            tokens_out_name="t5_tokens",
            mask_out_name="t5_tokens_mask",
            tokenizer=model.t5_tokenizer,
            max_token_length=PROMPT_MAX_LENGTH,
        )
        encode_prompt = EncodeAnimaText(
            tokens_name="tokens",
            tokens_attention_mask_name="tokens_mask",
            t5_tokens_name="t5_tokens",
            t5_tokens_attention_mask_name="t5_tokens_mask",
            hidden_state_out_name="text_encoder_hidden_state",
            text_encoder=model.text_encoder,
            text_conditioner=model.text_conditioner,
            autocast_contexts=[model.text_encoder_autocast_context],
            dtype=model.text_encoder_train_dtype.torch_dtype(),
        )

        modules = [rescale_image, tokenize_prompt, tokenize_t5]
        if not config.train_text_encoder_or_embedding():
            modules.append(encode_prompt)
        return modules

    def _cache_modules(self, config: TrainConfig, model: AnimaPixelModel, model_setup: BaseAnimaPixelSetup):
        image_split_names = ["pixel_image", "original_resolution", "crop_offset"]
        image_aggregate_names = ["crop_resolution", "image_path"]
        text_split_names = []
        sort_names = image_aggregate_names + image_split_names + [
            "prompt", "tokens", "tokens_mask", "t5_tokens", "t5_tokens_mask", "text_encoder_hidden_state", "concept",
        ]
        if not config.train_text_encoder_or_embedding():
            text_split_names += ["tokens", "tokens_mask", "t5_tokens", "t5_tokens_mask", "text_encoder_hidden_state"]

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
            "t5_tokens",
            "t5_tokens_mask",
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
        save_image = SaveImage(
            image_in_name="pixel_image",
            original_path_in_name="image_path",
            path=debug_dir,
            in_range_min=-1,
            in_range_max=1,
        )
        decode_prompt = DecodeTokens(in_name="tokens", out_name="decoded_prompt", tokenizer=model.tokenizer)
        save_prompt = SaveText(text_in_name="decoded_prompt", original_path_in_name="image_path", path=debug_dir)
        return [save_image, decode_prompt, save_prompt]

    def _create_dataset(
            self,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        return DataLoaderText2ImageMixin._create_dataset(
            self,
            config,
            model,
            model_setup,
            train_progress,
            is_validation,
            # Keep 16x16 pixel patches aligned for compiled transformer blocks.
            aspect_bucketing_quantization=64,
            allow_video_files=False,
            vae_frame_dim=False,
            supports_inpainting=False,
        )

    def _load_input_modules(self, config: TrainConfig, train_dtype, vae_frame_dim: bool = False) -> list:
        modules = DataLoaderText2ImageMixin._load_input_modules(self, config, train_dtype, vae_frame_dim)
        modules.append(NormalizeImageRange("image"))
        return modules


factory.register(BaseDataLoader, AnimaPixelBaseDataLoader, ModelType.ANIMA_PIXEL)
