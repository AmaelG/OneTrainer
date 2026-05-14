from modules.model.NanosaurModel import NanosaurModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.modelSaver.nanosaur.NanosaurModelSaver import NanosaurModelSaver
from modules.util import factory
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch


class NanosaurFineTuneModelSaver(BaseModelSaver, InternalModelSaverMixin):
    def save(self, model: NanosaurModel, model_type: ModelType, output_model_format: ModelFormat, output_model_destination: str, dtype: torch.dtype | None):
        del model_type
        NanosaurModelSaver().save(model, output_model_format, output_model_destination, dtype)
        if output_model_format == ModelFormat.INTERNAL:
            self._save_internal_data(model, output_model_destination)


factory.register(BaseModelSaver, NanosaurFineTuneModelSaver, ModelType.NANOSAUR, TrainingMethod.FINE_TUNE)
