import os
import traceback

from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import (
    AnimaPipeline,
    AnimaTransformer3DModel,
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import AutoTokenizer, Qwen3Model


class AnimaModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model,
                model_type,
                weight_dtypes,
                base_model_name,
                transformer_model_name,
                vae_model_name,
                quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if transformer_model_name:
            raise NotImplementedError("Anima transformer override is not implemented yet")

        diffusers_sub = ["transformer"]
        if not vae_model_name:
            diffusers_sub.append("vae")

        self._prepare_sub_modules(
            base_model_name,
            diffusers_modules=diffusers_sub,
            transformers_modules=["text_encoder"],
        )

        qwen_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )
        t5_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer_2",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            Qwen3Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
                quantization=quantization,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
                quantization,
            )

        transformer = self._load_diffusers_sub_module(
            AnimaTransformer3DModel,
            weight_dtypes.transformer,
            weight_dtypes.train_dtype,
            base_model_name,
            "transformer",
            quantization,
        )

        model.model_type = model_type
        model.qwen_tokenizer = qwen_tokenizer
        model.t5_tokenizer = t5_tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer
        model.ensure_tokenizer_padding()

    def __load_safetensors(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if transformer_model_name:
            raise NotImplementedError("Anima single-file transformer override is not implemented yet")

        pipeline = AnimaPipeline.from_single_file(base_model_name)

        qwen_tokenizer = getattr(pipeline, "tokenizer", None)
        t5_tokenizer = getattr(pipeline, "tokenizer_2", None)
        if qwen_tokenizer is None or t5_tokenizer is None:
            raise RuntimeError("Anima tokenizers are missing from loaded pipeline")

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
                quantization=quantization,
            )
        else:
            vae = self._convert_diffusers_sub_module_to_dtype(
                pipeline.vae,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                quantization,
            )

        text_encoder = self._convert_transformers_sub_module_to_dtype(
            pipeline.text_encoder,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
        )
        transformer = self._convert_diffusers_sub_module_to_dtype(
            pipeline.transformer,
            weight_dtypes.transformer,
            weight_dtypes.train_dtype,
            quantization,
        )

        model.model_type = model_type
        model.qwen_tokenizer = qwen_tokenizer
        model.t5_tokenizer = t5_tokenizer
        model.noise_scheduler = pipeline.scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer
        model.ensure_tokenizer_padding()

    def load(
            self,
            model: AnimaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model,
                model_type,
                weight_dtypes,
                model_names.base_model,
                model_names.transformer_model,
                model_names.vae_model,
                quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model,
                model_type,
                weight_dtypes,
                model_names.base_model,
                model_names.transformer_model,
                model_names.vae_model,
                quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model,
                model_type,
                weight_dtypes,
                model_names.base_model,
                model_names.transformer_model,
                model_names.vae_model,
                quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
