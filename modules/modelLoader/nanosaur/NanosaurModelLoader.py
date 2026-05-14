import os
import sys
from pathlib import Path

from modules.model.NanosaurModel import NanosaurModel, _NANOSAUR_SOURCE_DIR
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch
from safetensors.torch import load_file

if str(_NANOSAUR_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_NANOSAUR_SOURCE_DIR))

from model_lora import (  # noqa: E402
    LoraNanoSaurTransformer2DModel,
    VAE_LATENT_DIM,
    _clean_state_dict,
    build_text_encoder,
)
from vae import NanoSaurVAE  # noqa: E402


class NanosaurModelLoader(BaseModelLoader, InternalModelLoaderMixin):
    TRANSFORMER_FILENAME = "transformer.safetensor"
    TEXT_ENCODER_FILENAME = "text_encoder.safetensor"
    VAE_FILENAME = "vae.safetensor"

    def _checkpoint_path(self, model_dir: str, filename: str) -> Path:
        path = Path(model_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing Nanosaur checkpoint: {path}")
        return path

    def _load_components(
            self,
            model: NanosaurModel,
            model_type: ModelType,
            model_dir: str,
            weight_dtypes: ModelWeightDtypes,
    ):
        train_dtype = weight_dtypes.train_dtype.torch_dtype() or torch.float32
        transformer_dtype = weight_dtypes.transformer.torch_dtype() or train_dtype
        text_encoder_dtype = weight_dtypes.text_encoder.torch_dtype() or train_dtype
        vae_dtype = weight_dtypes.vae.torch_dtype() or train_dtype

        transformer_path = self._checkpoint_path(model_dir, self.TRANSFORMER_FILENAME)
        text_encoder_path = self._checkpoint_path(model_dir, self.TEXT_ENCODER_FILENAME)
        vae_path = self._checkpoint_path(model_dir, self.VAE_FILENAME)

        transformer = LoraNanoSaurTransformer2DModel()
        transformer.load_state_dict(_clean_state_dict(load_file(transformer_path, device="cpu")), strict=True)
        transformer = transformer.to(dtype=transformer_dtype)

        tokenizer, text_encoder = build_text_encoder("cpu", text_encoder_dtype, text_encoder_path)
        tokenizer.spiece_model = load_file(text_encoder_path, device="cpu")["spiece_model"]

        vae = NanoSaurVAE(latent_dim=VAE_LATENT_DIM)
        vae.load_state_dict(_clean_state_dict(load_file(vae_path, device="cpu")), strict=True)
        vae = vae.to(dtype=vae_dtype)

        model.model_type = model_type
        model.transformer = transformer
        model.tokenizer = tokenizer
        model.text_encoder = text_encoder
        model.vae = vae
        model.train_dtype = weight_dtypes.train_dtype

    def load(
            self,
            model: NanosaurModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        del quantization

        model_dir = model_names.base_model
        if os.path.exists(os.path.join(model_dir, "meta.json")):
            self._load_internal_data(model, model_dir)

        self._load_components(model, model_type, model_dir, weight_dtypes)
