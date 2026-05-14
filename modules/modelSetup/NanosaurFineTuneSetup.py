from modules.model.NanosaurModel import NanosaurModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseNanosaurSetup import BaseNanosaurSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class NanosaurFineTuneSetup(BaseNanosaurSetup):
    def create_parameters(self, model: NanosaurModel, config: TrainConfig) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        self._create_model_part_parameters(parameter_group_collection, "text_encoder", model.text_encoder, config.text_encoder)
        self._create_model_part_parameters(
            parameter_group_collection, "transformer", model.transformer, config.transformer,
            freeze=ModuleFilter.create(config), debug=config.debug_mode,
        )
        return parameter_group_collection

    def __setup_requires_grad(self, model: NanosaurModel, config: TrainConfig):
        self._setup_model_part_requires_grad("text_encoder", model.text_encoder, config.text_encoder, model.train_progress)
        self._setup_model_part_requires_grad("transformer", model.transformer, config.transformer, model.train_progress)
        model.vae.requires_grad_(False)

    def setup_model(self, model: NanosaurModel, config: TrainConfig):
        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(self, model: NanosaurModel, config: TrainConfig):
        model.vae_to(self.train_device if not config.latent_caching else self.temp_device)
        model.text_encoder_to(self.train_device if config.train_text_encoder_or_embedding() or not config.latent_caching else self.temp_device)
        model.transformer_to(self.train_device)
        model.vae.eval()
        model.text_encoder.train(config.text_encoder.train)
        model.transformer.train(config.transformer.train)

    def after_optimizer_step(self, model: NanosaurModel, config: TrainConfig, train_progress: TrainProgress):
        self.__setup_requires_grad(model, config)


factory.register(BaseModelSetup, NanosaurFineTuneSetup, ModelType.NANOSAUR, TrainingMethod.FINE_TUNE)
