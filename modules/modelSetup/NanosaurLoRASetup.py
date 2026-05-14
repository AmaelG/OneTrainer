from modules.model.NanosaurModel import NanosaurModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseNanosaurSetup import BaseNanosaurSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn


class NanosaurLoRASetup(BaseNanosaurSetup):
    DEFAULT_LORA_FILTER = ["blocks.", "text_refine_blocks.", "dec_net.res_blocks."]

    def __matched_layer_filter(self, model: NanosaurModel, config: TrainConfig) -> list[str]:
        layer_filter = [x.strip() for x in config.layer_filter.split(",") if x.strip()]
        if not layer_filter:
            return self.DEFAULT_LORA_FILTER

        matched_filter = []
        matched_names = set()
        linear_or_conv_names = [
            name.replace(".checkpoint.", ".")
            for name, module in model.transformer.named_modules()
            if isinstance(module, nn.Linear | nn.Conv2d)
        ]

        for pattern in layer_filter:
            module_filter = ModuleFilter(pattern, use_regex=config.layer_filter_regex)
            pattern_matches = {name for name in linear_or_conv_names if module_filter.matches(name)}
            # LoRAModuleWrapper rejects filters that do not get used. Since matching uses any(...),
            # overlapping filters after a broader one are never marked used. Keep only filters that
            # add at least one new module while preserving the final selected layer set.
            if pattern_matches - matched_names:
                matched_filter.append(pattern)
                matched_names.update(pattern_matches)

        if len(matched_filter) != len(layer_filter):
            unmatched = [pattern for pattern in layer_filter if pattern not in matched_filter]
            print(f"Ignoring redundant or unmatched Nanosaur LoRA layer filters: {unmatched}")

        return matched_filter or self.DEFAULT_LORA_FILTER

    def create_parameters(self, model: NanosaurModel, config: TrainConfig) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        self._create_model_part_parameters(parameter_group_collection, "transformer_lora", model.transformer_lora, config.transformer)
        return parameter_group_collection

    def __setup_requires_grad(self, model: NanosaurModel, config: TrainConfig):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
        self._setup_model_part_requires_grad("transformer_lora", model.transformer_lora, config.transformer, model.train_progress)

    def setup_model(self, model: NanosaurModel, config: TrainConfig):
        layer_filter = self.__matched_layer_filter(model, config)
        model.transformer_lora = LoRAModuleWrapper(model.transformer, "lora_transformer", config, layer_filter)
        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(self, model: NanosaurModel, config: TrainConfig):
        model.vae_to(self.train_device if not config.latent_caching else self.temp_device)
        model.text_encoder_to(self.temp_device if config.latent_caching else self.train_device)
        model.transformer_to(self.train_device)
        model.vae.eval()
        model.text_encoder.eval()
        model.transformer.train(config.transformer.train)

    def after_optimizer_step(self, model: NanosaurModel, config: TrainConfig, train_progress: TrainProgress):
        self.__setup_requires_grad(model, config)


factory.register(BaseModelSetup, NanosaurLoRASetup, ModelType.NANOSAUR, TrainingMethod.LORA)
