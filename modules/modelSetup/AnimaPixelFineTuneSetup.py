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
            filters = self.train_filters(config)
            transformer_parameters = []
            selected = []
            deselected = []
            for name, param in model.transformer.named_parameters():
                if param.is_floating_point() and self.matches_filter(name, "transformer", filters):
                    transformer_parameters.append(param)
                    selected.append(name)
                else:
                    deselected.append(name)

            detailer_parameters = []
            detailer_selected = []
            detailer_deselected = []
            for name, param in model.detailer_head.named_parameters():
                if param.is_floating_point() and self.matches_filter(name, "detailer_head", filters):
                    detailer_parameters.append(param)
                    detailer_selected.append(name)
                else:
                    detailer_deselected.append(name)

            print(f"Selected ANIMA_PIXEL transformer layers: {len(selected)}")
            print(f"Deselected ANIMA_PIXEL transformer layers: {len(deselected)}")
            print(f"Selected ANIMA_PIXEL detailer layers: {len(detailer_selected)}")
            print(f"Deselected ANIMA_PIXEL detailer layers: {len(detailer_deselected)}")
            if self.debug_mode:
                print("Selected ANIMA_PIXEL transformer layer names:")
                for name in selected:
                    print(f"  {name}")
                print("Selected ANIMA_PIXEL detailer layer names:")
                for name in detailer_selected:
                    print(f"  {name}")
            else:
                print("Note: Enable Debug mode to see the full list of selected layer names")

            if transformer_parameters:
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name="transformer",
                    parameters=transformer_parameters,
                    learning_rate=config.transformer.learning_rate,
                ))
            if detailer_parameters:
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name="detailer_head",
                    parameters=detailer_parameters,
                    learning_rate=config.transformer.learning_rate,
                ))

        if config.train_any_embedding() or config.train_any_output_embedding():
            raise NotImplementedError("Embeddings not implemented for Anima Pixel")

        return parameter_group_collection

    def __setup_requires_grad(self, model: AnimaPixelModel, config: TrainConfig):
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)
            model.text_conditioner.requires_grad_(False)

        model.transformer.requires_grad_(False)
        model.detailer_head.requires_grad_(False)

        if config.transformer.train:
            filters = self.train_filters(config)
            for name, param in model.transformer.named_parameters():
                if param.is_floating_point() and self.matches_filter(name, "transformer", filters):
                    param.requires_grad_(True)
            for name, param in model.detailer_head.named_parameters():
                if param.is_floating_point() and self.matches_filter(name, "detailer_head", filters):
                    param.requires_grad_(True)

    def setup_model(self, model: AnimaPixelModel, config: TrainConfig):
        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(self, model: AnimaPixelModel, config: TrainConfig):
        text_encoder_on_train_device = not config.text_caching
        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)
        model.detailer_head_to(self.train_device)

        model.text_encoder.eval()
        model.text_conditioner.eval()
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
