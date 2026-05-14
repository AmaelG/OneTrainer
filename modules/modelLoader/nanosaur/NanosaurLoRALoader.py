from modules.model.NanosaurModel import NanosaurModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin


class NanosaurLoRALoader(LoRALoaderMixin):
    def _get_convert_key_sets(self, model: NanosaurModel):
        return None

    def load(self, model: NanosaurModel, model_names):
        self._load(model, model_names)
