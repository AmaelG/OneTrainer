from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.anima.AnimaModelLoader import AnimaModelLoader
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.util.enum.ModelType import ModelType


AnimaFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.ANIMA: "resources/sd_model_spec/anima.json"},
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
)
