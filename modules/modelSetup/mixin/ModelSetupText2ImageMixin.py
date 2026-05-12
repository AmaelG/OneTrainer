from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

import torch


class ModelSetupText2ImageMixin(metaclass=ABCMeta):
    @abstractmethod
    def prepare_text_caching(self, model: BaseModel, config: TrainConfig):
        pass

    @torch.no_grad()
    def calculate_validation_losses(
            self,
            model: BaseModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
    ) -> dict:
        losses = {}
        timesteps = [int(timestep.strip()) for timestep in config.validation_timesteps.split(",") if timestep.strip()]

        if not timesteps:
            model_output_data = self.predict(model, batch, config, train_progress, deterministic=True)
            loss = self.calculate_loss(model, batch, model_output_data, config)
            return {None: loss.item()}

        for timestep_value in timesteps:
            timestep = torch.full(
                (batch["latent_image"].shape[0],),
                timestep_value,
                dtype=torch.long,
                device=config.train_device,
            )
            model_output_data = self.predict(
                model,
                batch,
                config,
                train_progress,
                deterministic=True,
                timestep=timestep,
            )
            loss = self.calculate_loss(model, batch, model_output_data, config)
            losses[timestep_value] = loss.item()

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
