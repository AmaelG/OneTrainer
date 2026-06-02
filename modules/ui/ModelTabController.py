

from modules.util import create
from modules.util.config.TrainConfig import TrainConfig


class ModelTabController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def get_presets(self) -> dict:
        cls = create.get_model_setup_class(self.train_config.model_type, self.train_config.training_method)
        return cls.LAYER_PRESETS if cls is not None else {"full": []}

    def get_quantization_presets(self) -> dict:
        cls = create.get_model_setup_class(self.train_config.model_type, self.train_config.training_method)
        return getattr(cls, "QUANTIZATION_LAYER_PRESETS", {"full": []}) if cls is not None else {"full": []}
