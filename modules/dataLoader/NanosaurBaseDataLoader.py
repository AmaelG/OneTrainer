import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.nanosaur.DecodeNanoSaurVAE import DecodeNanoSaurVAE
from modules.dataLoader.nanosaur.EncodeNanoSaurText import EncodeNanoSaurText
from modules.dataLoader.nanosaur.EncodeNanoSaurVAE import EncodeNanoSaurVAE
from modules.model.BaseModel import BaseModel
from modules.model.NanosaurModel import NanosaurModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseNanosaurSetup import BaseNanosaurSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText


class NanosaurBaseDataLoader(BaseDataLoader, DataLoaderText2ImageMixin):
    def _preparation_modules(self, config: TrainConfig, model: NanosaurModel):
        rescale_image = RescaleImageChannels(
            image_in_name="image", image_out_name="image",
            in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1,
        )
        encode_image = EncodeNanoSaurVAE(
            in_name="image", out_name="latent_image", model=model,
            autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(),
        )
        encode_text = EncodeNanoSaurText(
            in_name="prompt", out_name="text_embedding", model=model,
            autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(),
        )

        modules = [rescale_image, encode_image]
        if not config.train_text_encoder_or_embedding():
            modules.append(encode_text)
        return modules

    def _cache_modules(self, config: TrainConfig, model: NanosaurModel, model_setup: BaseNanosaurSetup):
        image_split_names = ["latent_image", "original_resolution", "crop_offset"]
        image_aggregate_names = ["crop_resolution", "image_path"]
        text_split_names = [] if config.train_text_encoder_or_embedding() else ["text_embedding"]
        sort_names = image_aggregate_names + image_split_names + ["prompt", "concept"] + text_split_names
        return self._cache_modules_from_names(
            model, model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=not config.train_text_encoder_or_embedding(),
        )

    def _output_modules(self, config: TrainConfig, model: NanosaurModel, model_setup: BaseNanosaurSetup):
        output_names = [
            "image_path", "latent_image", "prompt",
            "original_resolution", "crop_resolution", "crop_offset",
        ]
        if not config.train_text_encoder_or_embedding():
            output_names.append("text_embedding")

        return self._output_modules_from_out_names(
            model, model_setup,
            output_names=output_names,
            config=config,
            vae=None,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: NanosaurModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.vae_to(self.train_device)

        decode_image = DecodeNanoSaurVAE(
            in_name="latent_image", out_name="decoded_image", model=model,
            autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(),
        )
        save_image = SaveImage(
            image_in_name="decoded_image", original_path_in_name="image_path", path=debug_dir,
            in_range_min=-1, in_range_max=1, before_save_fun=before_save_fun,
        )
        save_prompt = SaveText(text_in_name="prompt", original_path_in_name="image_path", path=debug_dir)
        return [decode_image, save_image, save_prompt]

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
        mask_augmentation = self._mask_augmentation_modules(config)
        aspect_bucketing_in = self._aspect_bucketing_in(config, 16, False)
        crop_modules = self._crop_modules(config)
        augmentation_modules = self._augmentation_modules(config)
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
                preparation_modules,
                cache_modules,
                debug_modules if config.debug_mode else None,
                output_modules,
            ],
            train_progress,
            is_validation,
        )


factory.register(BaseDataLoader, NanosaurBaseDataLoader, ModelType.NANOSAUR)
