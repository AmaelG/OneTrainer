from modules.model.AnimaPixelModel import AnimaPixelModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.anima.AnimaPixelModelSaver import AnimaPixelModelSaver
from modules.util.enum.ModelType import ModelType


AnimaPixelFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.ANIMA_PIXEL,
    model_class=AnimaPixelModel,
    model_saver_class=AnimaPixelModelSaver,
    embedding_saver_class=None,
)
