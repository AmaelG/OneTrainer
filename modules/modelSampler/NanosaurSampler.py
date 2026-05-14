from collections.abc import Callable

from modules.model.NanosaurModel import NanosaurModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


class NanosaurSampler(BaseModelSampler):
    SAMPLE_SHIFT = 4.0

    def __init__(self, train_device: torch.device, temp_device: torch.device, model: NanosaurModel, model_type: ModelType):
        super().__init__(train_device, temp_device)
        self.model = model
        self.model_type = model_type

    def _sampling_timesteps(self, steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)[:-1]
        return self.SAMPLE_SHIFT * timesteps / (1 + (self.SAMPLE_SHIFT - 1) * timesteps)

    def _sample_latents(
            self,
            z: torch.Tensor,
            cond: torch.Tensor,
            null_cond: torch.Tensor,
            steps: int,
            guidance_scale: float,
            on_update_progress: Callable[[int, int], None],
    ) -> torch.Tensor:
        latents = z.clone()
        batch = latents.size(0)
        dtype = latents.dtype
        latent_shape = [1] + [1] * (latents.dim() - 1)
        timesteps = self._sampling_timesteps(steps, latents.device, dtype)
        momentum = None

        for index, t_curr in enumerate(tqdm(timesteps, desc="sampling")):
            t_next = timesteps[index + 1] if index + 1 < steps else torch.tensor(0.0, device=latents.device, dtype=dtype)
            dt = t_curr - t_next
            t = t_curr.expand(batch)
            guided_output, _ = self.model.transformer(latents, t, cond, return_x0=True)
            guided = (latents - guided_output) / t_curr
            step_fraction = index / steps
            if 0.03 < step_fraction < 0.8:
                step_uncond = index % 2 == 1
                unguided_output, _ = self.model.transformer(latents, t, null_cond, uncond=step_uncond, return_x0=True)
                unguided = (latents - unguided_output) / t_curr
                guided = unguided + guidance_scale * (guided - unguided)
                if momentum is None:
                    momentum = guided.clone()
                effective_velocity = guided + 0.5 * (guided - momentum)
                momentum = 0.4 * guided + 0.6 * momentum
                guided = effective_velocity
            latents = latents - dt.view(latent_shape) * guided
            on_update_progress(index + 1, steps)

        return latents

    @torch.no_grad()
    def __sample(self, sample_config: SampleConfig, on_update_progress: Callable[[int, int], None]) -> ModelSamplerOutput:
        generator = torch.Generator(device=self.train_device)
        if sample_config.random_seed:
            generator.seed()
        else:
            generator.manual_seed(sample_config.seed)

        height = self.quantize_resolution(sample_config.height, 16)
        width = self.quantize_resolution(sample_config.width, 16)
        latent_height = height // 16
        latent_width = width // 16

        self.model.to(self.train_device)
        self.model.eval()
        with self.model.autocast_context:
            train_dtype = self.model.train_dtype.torch_dtype() or torch.float32
            cond = self.model.encode_text(sample_config.prompt, self.train_device).to(dtype=train_dtype)
            negative_prompt = sample_config.negative_prompt or ""
            null_cond = self.model.encode_text(negative_prompt, self.train_device).to(dtype=train_dtype)
            noise = torch.randn((1, 96, latent_height, latent_width), generator=generator, device=self.train_device, dtype=train_dtype)
            sampled = self._sample_latents(noise, cond, null_cond, sample_config.diffusion_steps, sample_config.cfg_scale, on_update_progress)
            image = self.model.decode_vae(sampled)[0].clamp(-1, 1)

        self.model.to(self.temp_device)
        torch_gc()
        image = ((image.detach().cpu().float() + 1.0) / 2.0).clamp(0, 1)
        return ModelSamplerOutput(FileType.IMAGE, to_pil_image(image))

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


factory.register(BaseModelSampler, NanosaurSampler, ModelType.NANOSAUR)
