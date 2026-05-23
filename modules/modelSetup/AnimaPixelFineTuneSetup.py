import torch

from modules.model.AnimaPixelModel import AnimaPixelModel
from modules.modelSetup.BaseAnimaPixelSetup import BaseAnimaPixelSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress


class AnimaPixelFineTuneSetup(BaseAnimaPixelSetup):
    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super().__init__(train_device=train_device, temp_device=temp_device, debug_mode=debug_mode)

    def create_parameters(self, model: AnimaPixelModel, config: TrainConfig) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        if config.transformer.train:
            filters = self.trainable_transformer_filters()
            selected = []
            deselected = []
            transformer_parameters = []
            for name, param in model.transformer.named_parameters():
                if any(f.matches(name) for f in filters) and param.is_floating_point():
                    selected.append(name)
                    transformer_parameters.append(param)
                else:
                    deselected.append(name)

            print(f"Selected layers: {len(selected)}")
            print(f"Deselected layers: {len(deselected)}")
            print("Selected layer names:")
            for name in selected:
                print(f"  {name}")

            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="transformer",
                parameters=transformer_parameters,
                learning_rate=config.transformer.learning_rate,
            ))
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="detailer_head",
                parameters=[p for p in model.detailer_head.parameters() if p.is_floating_point()],
                learning_rate=config.transformer.learning_rate,
            ))
        return parameter_group_collection

    def __setup_requires_grad(self, model: AnimaPixelModel, config: TrainConfig):
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)

        for param in model.transformer.parameters():
            if param.is_floating_point():
                param.requires_grad_(False)
        for param in model.detailer_head.parameters():
            if param.is_floating_point():
                param.requires_grad_(False)

        if config.transformer.train:
            filters = self.trainable_transformer_filters()
            for name, param in model.transformer.named_parameters():
                if param.is_floating_point() and any(f.matches(name) for f in filters):
                    param.requires_grad_(True)
            for param in model.detailer_head.parameters():
                if param.is_floating_point():
                    param.requires_grad_(True)

    def setup_model(self, model: AnimaPixelModel, config: TrainConfig):
        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(self, model: AnimaPixelModel, config: TrainConfig):
        text_encoder_on_train_device = config.train_text_encoder_or_embedding() or not config.latent_caching
        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)
        model.detailer_head_to(self.train_device)

        if model.text_encoder is not None:
            model.text_encoder.eval()
        if config.transformer.train:
            model.transformer.train()
            model.detailer_head.train()
        else:
            model.transformer.eval()
            model.detailer_head.eval()

    def after_optimizer_step(self, model: AnimaPixelModel, config: TrainConfig, train_progress: TrainProgress):
        del train_progress
        self.__setup_requires_grad(model, config)


factory.register(BaseModelSetup, AnimaPixelFineTuneSetup, ModelType.ANIMA_PIXEL, TrainingMethod.FINE_TUNE)
