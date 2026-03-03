from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import AnimaPipeline, AutoencoderKLQwenImage, DiffusionPipeline
from transformers import PreTrainedModel, PreTrainedTokenizer

class AnimaModel(BaseModel):
    qwen_tokenizer: PreTrainedTokenizer | None
    t5_tokenizer: PreTrainedTokenizer | None
    noise_scheduler: object | None
    text_encoder: PreTrainedModel | None
    vae: AutoencoderKLQwenImage | None
    transformer: object | None

    text_encoder_autocast_context: torch.autocast | nullcontext
    text_encoder_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.qwen_tokenizer = None
        self.t5_tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()
        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.transformer_lora,
        ] if a is not None]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        return AnimaPipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.qwen_tokenizer,
            tokenizer_2=self.t5_tokenizer,
        )

    def __tokenize_text(
            self,
            text: list[str],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.qwen_tokenizer is None or self.t5_tokenizer is None:
            raise RuntimeError("Anima tokenizers are not initialized")

        prompt_max_length = self.prompt_max_length()

        qwen_tokenizer_output = self.qwen_tokenizer(
            text,
            max_length=prompt_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        t5_tokenizer_output = self.t5_tokenizer(
            text,
            max_length=prompt_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        qwen_tokens = qwen_tokenizer_output.input_ids
        qwen_tokens_mask = qwen_tokenizer_output.attention_mask
        t5_tokens = t5_tokenizer_output.input_ids
        t5_tokens_mask = t5_tokenizer_output.attention_mask

        return qwen_tokens, qwen_tokens_mask, t5_tokens, t5_tokens_mask

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] | None = None,
            tokens: Tensor | None = None,
            tokens_mask: Tensor | None = None,
            t5_tokens: Tensor | None = None,
            t5_tokens_mask: Tensor | None = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor | None = None,
    ) -> Tensor:
        del batch_size, rand

        if tokens is None and text is not None:
            prompt_list = [text] if isinstance(text, str) else list(text)
            tokens, tokens_mask, t5_tokens, t5_tokens_mask = self.__tokenize_text(prompt_list)

        encoder_device = train_device
        if self.text_encoder is not None:
            encoder_device = self.text_encoder.device

        if tokens is not None:
            tokens = tokens.to(encoder_device)
        if tokens_mask is None and tokens is not None:
            tokens_mask = torch.ones_like(tokens, device=encoder_device)
        elif tokens_mask is not None:
            tokens_mask = tokens_mask.to(encoder_device)

        if t5_tokens is not None:
            t5_tokens = t5_tokens.to(encoder_device)
        if t5_tokens_mask is None and t5_tokens is not None:
            t5_tokens_mask = torch.ones_like(t5_tokens, device=encoder_device)
        elif t5_tokens_mask is not None:
            t5_tokens_mask = t5_tokens_mask.to(encoder_device)

        if text_encoder_output is None:
            if self.text_encoder is None:
                raise RuntimeError("text_encoder_output is required when no text_encoder is present.")
            if tokens is None:
                raise RuntimeError("Anima text encoding requires Qwen tokens when text_encoder_output is not provided.")

            with self.text_encoder_autocast_context:
                text_encoder_output = self.text_encoder(
                    input_ids=tokens,
                    attention_mask=tokens_mask,
                )

                if isinstance(text_encoder_output, tuple):
                    text_encoder_output = text_encoder_output[0]
                else:
                    text_encoder_output = text_encoder_output.last_hidden_state

        if t5_tokens is None:
            raise RuntimeError("Anima text encoding requires T5 token tensors.")

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError

        prompt_max_length = self.prompt_max_length()
        t5_weights = t5_tokens_mask.to(dtype=torch.float32).unsqueeze(-1)

        condition = self.transformer.preprocess_text_embeds(
            text_encoder_output.to(dtype=self.train_dtype.torch_dtype()),
            t5_tokens.to(dtype=torch.int32),
            t5xxl_weights=t5_weights,
        )

        if condition.shape[1] < prompt_max_length:
            condition = torch.nn.functional.pad(condition, (0, 0, 0, prompt_max_length - condition.shape[1]))
        else:
            condition = condition[:, :prompt_max_length, :]

        return condition

    def scale_latents(self, latents: Tensor) -> Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return (latents - latents_mean) * latents_std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return latents / latents_std + latents_mean

    def ensure_tokenizer_padding(self):
        if self.qwen_tokenizer is not None and self.qwen_tokenizer.pad_token_id is None:
            if self.qwen_tokenizer.eos_token is not None:
                self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
            elif self.qwen_tokenizer.unk_token_id is not None:
                self.qwen_tokenizer.pad_token_id = self.qwen_tokenizer.unk_token_id

    def prompt_max_length(self) -> int:
        transformer_config = getattr(self.transformer, "config", None)
        prompt_max_length = getattr(transformer_config, "adapter_sequence_length", None)
        if not isinstance(prompt_max_length, int) or prompt_max_length <= 0:
            raise RuntimeError("Anima transformer adapter_sequence_length is not available")
        return prompt_max_length
