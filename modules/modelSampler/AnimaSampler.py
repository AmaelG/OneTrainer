import copy
from collections.abc import Callable

from modules.model.AnimaModel import AnimaModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.image_util import load_image
from modules.util.torch_util import torch_gc

import torch


class AnimaSampler(BaseModelSampler):
    def __init__(
        self,
        train_device: torch.device,
        temp_device: torch.device,
        model: AnimaModel,
        model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.pipeline = model.create_pipeline()

    def __activate_sampling_compiled_calls(self) -> list[torch.nn.Module]:
        swapped_modules = []
        for module in self.model.transformer.modules():
            compiled_call_impl = getattr(module, "_compiled_call_impl", None)
            sampling_compiled_call_impl = getattr(module, "_sampling_compiled_call_impl", None)
            if compiled_call_impl is None and sampling_compiled_call_impl is None:
                continue

            module._training_compiled_call_impl = compiled_call_impl
            module._compiled_call_impl = sampling_compiled_call_impl
            if module._compiled_call_impl is None:
                module.compile(fullgraph=True)
            swapped_modules.append(module)

        return swapped_modules

    def __restore_training_compiled_calls(self, swapped_modules: list[torch.nn.Module]):
        for module in swapped_modules:
            module._sampling_compiled_call_impl = getattr(module, "_compiled_call_impl", None)
            module._compiled_call_impl = getattr(module, "_training_compiled_call_impl", None)

    @torch.no_grad()
    def __sample(
        self,
        sample_config: SampleConfig,
        on_update_progress: Callable[[int, int], None],
    ) -> ModelSamplerOutput:
        generator = torch.Generator(device=self.train_device)
        if sample_config.random_seed:
            generator.seed()
        else:
            generator.manual_seed(sample_config.seed)

        height = self.quantize_resolution(sample_config.height, 16)
        width = self.quantize_resolution(sample_config.width, 16)

        image = None
        mask_image = None
        if sample_config.sample_inpainting:
            image = load_image(sample_config.base_image_path, convert_mode="RGB")
            mask_image = load_image(sample_config.mask_image_path, convert_mode="L")

        self.pipeline.scheduler = copy.deepcopy(self.model.noise_scheduler)
        self.model.to(self.train_device)
        self.pipeline.to(self.train_device)
        swapped_modules = self.__activate_sampling_compiled_calls()

        step_state = {"i": 0}

        def on_step_end(_pipeline, _step, _timestep, _kwargs):
            step_state["i"] += 1
            on_update_progress(step_state["i"], sample_config.diffusion_steps)
            if step_state["i"] >= _pipeline.num_timesteps:
                self.model.transformer_to(self.temp_device)
                self.model.text_encoder_to(self.temp_device)
                torch_gc()
            return None

        try:
            with self.model.autocast_context:
                result = self.pipeline(
                    prompt=sample_config.prompt,
                    negative_prompt=sample_config.negative_prompt,
                    image=image,
                    mask_image=mask_image,
                    strength=1.0,
                    width=width,
                    height=height,
                    num_inference_steps=sample_config.diffusion_steps,
                    guidance_scale=sample_config.cfg_scale,
                    generator=generator,
                    callback_on_step_end=on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
        finally:
            self.__restore_training_compiled_calls(swapped_modules)

        self.model.to(self.temp_device)
        torch_gc()

        return ModelSamplerOutput(
            file_type=FileType.IMAGE,
            data=result.images[0],
        )

    def sample(
        self,
        sample_config: SampleConfig,
        destination: str,
        image_format: ImageFormat | None = None,
        video_format: VideoFormat | None = None,
        audio_format: AudioFormat | None = None,
        on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
        on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        sampler_output = self.__sample(
            sample_config=sample_config,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output,
            destination,
            image_format,
            video_format,
            audio_format,
        )

        on_sample(sampler_output)


factory.register(BaseModelSampler, AnimaSampler, ModelType.ANIMA)
