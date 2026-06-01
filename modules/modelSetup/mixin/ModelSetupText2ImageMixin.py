from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class ModelSetupText2ImageMixin(metaclass=ABCMeta):
    @abstractmethod
    def prepare_text_caching(self, model: BaseModel, config: TrainConfig):
        pass

    @abstractmethod
    def predict(
            self,
            model: BaseModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
            timestep: Tensor | None = None,
    ) -> dict:
        pass

    @abstractmethod
    def calculate_loss(
            self,
            model: BaseModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        pass

    @torch.no_grad()
    def calculate_validation_losses(
            self,
            model: BaseModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
    ) -> dict[int, float]:
        losses = {}
        timesteps = [int(x.strip()) for x in config.validation_timesteps.split(',') if x.strip()]
        for validation_timestep in timesteps:
            timestep = torch.tensor(
                validation_timestep,
                dtype=torch.long,
                device=config.train_device,
            ).unsqueeze(0)
            model_output_data = self.predict(
                model, batch, config, train_progress,
                deterministic=True, timestep=timestep,
            )
            loss = self.calculate_loss(model, batch, model_output_data, config)
            losses[validation_timestep] = loss.item()
        return losses

    #for future use in samplers etc.
    '''@abstractmethod
    def prepare_training(self, model: BaseModel):
        pass

    @abstractmethod
    def prepare_text_inference(self, model: BaseModel):
        pass

    @abstractmethod
    def prepare_image_inference(self, model: BaseModel):
        pass'''
