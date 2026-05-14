from modules.model.NanosaurModel import NanosaurModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.nanosaur.NanosaurLoRALoader import NanosaurLoRALoader
from modules.modelLoader.nanosaur.NanosaurModelLoader import NanosaurModelLoader
from modules.util import factory
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class NanosaurLoRAModelLoader(BaseModelLoader, InternalModelLoaderMixin):
    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ) -> NanosaurModel:
        model = NanosaurModel(model_type=model_type)
        NanosaurModelLoader().load(model, model_type, model_names, weight_dtypes, quantization)
        self._load_internal_data(model, model_names.lora)
        NanosaurLoRALoader().load(model, model_names)
        return model


factory.register(BaseModelLoader, NanosaurLoRAModelLoader, ModelType.NANOSAUR, TrainingMethod.LORA)
