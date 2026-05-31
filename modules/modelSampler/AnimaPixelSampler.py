from collections.abc import Callable
import copy

import torch
from PIL import Image
from tqdm import tqdm

from modules.model.AnimaPixelModel import AnimaPixelModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc


class AnimaPixelSampler(BaseModelSampler):
    def __init__(self, train_device: torch.device, temp_device: torch.device, model: AnimaPixelModel, model_type: ModelType):
        super().__init__(train_device, temp_device)
        self.model = model
        self.model_type = model_type

    @staticmethod
    def __tensor_to_image(pixel_tensor: torch.Tensor) -> Image.Image:
        pixel_tensor = ((pixel_tensor[0].float().clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
        array = (pixel_tensor.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        return Image.fromarray(array)

    @torch.no_grad()
    def __sample(self, sample_config: SampleConfig, on_update_progress: Callable[[int, int], None]) -> ModelSamplerOutput:
        generator = torch.Generator(device=self.train_device)
        if sample_config.random_seed:
            generator.seed()
        else:
            generator.manual_seed(sample_config.seed)

        height = self.quantize_resolution(sample_config.height, 16)
        width = self.quantize_resolution(sample_config.width, 16)

        self.model.to(self.train_device)
        self.model.eval()
        noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
        noise_scheduler.set_timesteps(sample_config.diffusion_steps, device=self.train_device)

        with self.model.autocast_context:
            text_encoder_output = self.model.encode_text(
                train_device=self.train_device,
                text=sample_config.prompt,
            )
            negative_text_encoder_output = None
            if sample_config.cfg_scale > 1.0:
                negative_text_encoder_output = self.model.encode_text(
                    train_device=self.train_device,
                    text=sample_config.negative_prompt or "",
                )

            latents = torch.randn((1, 3, height, width), generator=generator, device=self.train_device, dtype=torch.float32)
            timesteps = noise_scheduler.timesteps
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                timestep_batch = timestep.expand(latents.shape[0]).to(device=self.train_device, dtype=torch.float32)
                model_timestep = timestep_batch / float(noise_scheduler.config.num_train_timesteps)
                flow_text = self.model.predict_pixel_flow(
                    noisy_image=latents.to(dtype=self.model.train_dtype.torch_dtype()),
                    timestep=model_timestep,
                    text_encoder_output=text_encoder_output,
                )
                if negative_text_encoder_output is not None:
                    flow_uncond = self.model.predict_pixel_flow(
                        noisy_image=latents.to(dtype=self.model.train_dtype.torch_dtype()),
                        timestep=model_timestep,
                        text_encoder_output=negative_text_encoder_output,
                    )
                    flow = flow_uncond + sample_config.cfg_scale * (flow_text - flow_uncond)
                else:
                    flow = flow_text

                latents = noise_scheduler.step(flow.float(), timestep, latents, return_dict=False)[0]
                on_update_progress(i + 1, sample_config.diffusion_steps)

        image = self.__tensor_to_image(latents)
        self.model.to(self.temp_device)
        torch_gc()
        return ModelSamplerOutput(file_type=FileType.IMAGE, data=image)

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
        sampler_output = self.__sample(sample_config, on_update_progress)
        self.save_sampler_output(sampler_output, destination, image_format, video_format, audio_format)
        on_sample(sampler_output)


factory.register(BaseModelSampler, AnimaPixelSampler, ModelType.ANIMA_PIXEL)
