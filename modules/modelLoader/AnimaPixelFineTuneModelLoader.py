from modules.model.AnimaPixelModel import AnimaPixelModel
from modules.modelLoader.anima.AnimaPixelModelLoader import AnimaPixelModelLoader
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.util.enum.ModelType import ModelType


AnimaPixelFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.ANIMA_PIXEL: "resources/sd_model_spec/anima_pixel.json"},
    model_class=AnimaPixelModel,
    model_loader_class=AnimaPixelModelLoader,
    embedding_loader_class=None,
)
