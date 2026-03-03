from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.anima.AnimaLoRALoader import AnimaLoRALoader
from modules.modelLoader.anima.AnimaModelLoader import AnimaModelLoader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.util.enum.ModelType import ModelType


AnimaLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.ANIMA: "resources/sd_model_spec/anima-lora.json"},
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
    lora_loader_class=AnimaLoRALoader,
)
