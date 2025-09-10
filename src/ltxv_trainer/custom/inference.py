from __future__ import annotations
from typing import *
from jaxtyping import Shaped, Bool, Int, Integer, Float

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, repeat

import inspect, functools, itertools
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum, auto

from diffusers.models.autoencoders import AutoencoderKLLTXVideo
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput, Transformer2DModelOutput
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_ltx import (
    apply_rotary_emb,
    LTXVideoRotaryPosEmbed,
    LTXAttention,
    LTXVideoAttnProcessor,
    LTXVideoTransformerBlock as _LTXVideoTransformerBlock,
    LTXVideoTransformer3DModel as _LTXVideoTransformer3DModel
)

from transformers import T5EncoderModel, T5Tokenizer, T5TokenizerFast
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

T = TypeVar('T')
def not_none(x: Optional[T]) -> T:
    assert x is not None
    return x

def linear_quadratic_schedule(num_steps: int, threshold_noise=0.025, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return torch.tensor([1.0])
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule[:-1])

def randn_like(x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    return torch.randn(
        size=x.size(),
        dtype=x.dtype,
        layout=x.layout,
        device=x.device,
        generator=generator,
    )

PrecomputedRotaryEmb = Tuple[Float[torch.Tensor, 'batch_size num_tokens inner_dim'], Float[torch.Tensor, 'batch_size num_tokens inner_dim']]

from diffusers import BitsAndBytesConfig
from transformers import (
    AutoModel,
    AutoProcessor,
    LlavaNextVideoForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

DEFAULT_VLM_CAPTION_INSTRUCTION = "Shortly describe the content of this video in two sentences."

class CaptionerType(str, Enum):
    """Enum for different types of video captioners."""

    LLAVA_NEXT_7B = "llava_next_7b"
    QWEN_25_VL = "qwen_25_vl"

def create_captioner(captioner_type: CaptionerType, **kwargs) -> "MediaCaptioningModel":
    """Factory function to create a video captioner.

    Args:
        captioner_type: The type of captioner to create
        **kwargs: Additional arguments to pass to the captioner constructor

    Returns:
        An instance of a MediaCaptioningModel
    """
    if captioner_type == CaptionerType.LLAVA_NEXT_7B:
        return TransformersVlmCaptioner(model_id="llava-hf/LLaVA-NeXT-Video-7B-hf", **kwargs)
    elif captioner_type == CaptionerType.QWEN_25_VL:
        return TransformersVlmCaptioner(model_id="Qwen/Qwen2.5-VL-7B-Instruct", **kwargs)
    else:
        raise ValueError(f"Unsupported captioner type: {captioner_type}")

class MediaCaptioningModel(ABC):
    """Abstract base class for video and image captioning models."""

    @abstractmethod
    def caption(self, path: Union[str, Path]) -> str:
        """Generate a caption for the given video or image.

        Args:
            path: Path to the video/image file to caption

        Returns:
            A string containing the generated caption
        """

    @staticmethod
    def _is_image_file(path: Union[str, Path]) -> bool:
        """Check if the file is an image based on extension."""
        return str(path).lower().endswith((".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp"))

    @staticmethod
    def _clean_raw_caption(caption: str) -> str:
        """Clean up the raw caption."""
        start = ["The", "This"]
        kind = ["video", "image", "scene", "animated sequence"]
        act = ["displays", "shows", "features", "depicts", "presents", "showcases", "captures"]

        for x, y, z in itertools.product(start, kind, act):
            caption = caption.replace(f"{x} {y} {z} ", "", 1)

        return caption

class TransformersVlmCaptioner(MediaCaptioningModel):
    """Video and image captioning model using models implemented in HuggingFace's `transformers`."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[Union[str, torch.device]] = None,
        use_8bit: bool = False,
        vlm_instruction: str = DEFAULT_VLM_CAPTION_INSTRUCTION,
    ):
        """Initialize the captioner.

        Args:
            model_id: HuggingFace model ID for LLaVA-NeXT-Video
            device: torch.device to use for the model
            use_8bit: Whether to use 8-bit quantization
            vlm_instruction: Instruction prompt for the model
        """
        self.device = torch.device(device if torch.cuda.is_available() and device is not None else "cpu")
        self.vlm_instruction = vlm_instruction
        self._load_model(model_id, use_8bit=use_8bit)

    def caption(
        self,
        path: Union[str, Path],
        fps: int = 3,
        clean_caption: bool = True,
    ) -> str:
        """Generate a caption for the given video or image.

        Args:
            path: Path to the video/image file to caption
            fps: Frames per second of the video, ignored for images.
            clean_caption: Whether to clean up the raw caption by removing common VLM patterns.

        Returns:
            A string containing the generated caption
        """
        # Determine if input is image or video
        is_image = self._is_image_file(path)

        # Prepare inputs
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.vlm_instruction},
                    {"type": "image" if is_image else "video", "path": str(path)},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            video_fps=fps if not is_image else None,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate caption
        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=None,
        )

        # Trim the generated tokens to exclude the input tokens
        output_tokens_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_tokens, strict=True)
        ]

        # Decode the generated tokens to text
        caption_raw = self.processor.batch_decode(
            output_tokens_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Clean up caption
        caption = self._clean_raw_caption(caption_raw) if clean_caption else caption_raw

        return caption

    def _load_model(self, model_id: str, use_8bit: bool) -> None:
        if model_id == "llava-hf/LLaVA-NeXT-Video-7B-hf":
            model_cls = LlavaNextVideoForConditionalGeneration
        elif model_id == "Qwen/Qwen2.5-VL-7B-Instruct":
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            model_cls = AutoModel

        quantization_config = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None
        self.model = model_cls.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            device_map=self.device.type,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=True,
            min_pixels=16 * 28 * 28,
            max_pixels=64 * 28 * 28,
        )

@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompts: Prompts, max_sequence_length: int = 256, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> TextEmbeds:
    if isinstance(prompts, list):
        tokenizer, text_encoder = not_none(tokenizer), not_none(text_encoder)
        device, dtype = (device or text_encoder.device), (dtype or text_encoder.dtype)
        text_inputs = tokenizer(
            prompts, padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        prompt_attention_mask = text_inputs.attention_mask.bool().to(text_encoder.device)
        prompt_embeds = text_encoder(
            text_inputs.input_ids.to(text_encoder.device),
            attention_mask=prompt_attention_mask
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return {"prompt_embeds": prompt_embeds, "prompt_attention_mask": prompt_attention_mask.to(device)}
    return {"prompt_embeds": prompts["prompt_embeds"].to(device=device, dtype=dtype), "prompt_attention_mask": prompts["prompt_attention_mask"].to(device=device)}

@torch.no_grad()
def encode_as_video(vae: AutoencoderKLLTXVideo, pixels: Float[torch.Tensor, 'b f c h w']) -> Float[torch.Tensor, 'b c latent_frames latent_height latent_width']:
    latents_mean, latents_std, scaling_factor = vae.latents_mean, vae.latents_std, vae.config['scaling_factor']
    latents_mean, latents_std = latents_mean.view(1, -1, 1, 1, 1), latents_std.view(1, -1, 1, 1, 1)
    latents = rearrange(pixels, 'b f c h w -> b c f h w').to(vae.dtype)
    latents = cast(AutoencoderKLOutput, vae.encode(latents*2-1, return_dict=True)).latent_dist.sample()
    return (latents - latents_mean) / latents_std * scaling_factor

@torch.no_grad()
def encode_as_images(vae: AutoencoderKLLTXVideo, pixels: Float[torch.Tensor, 'b f c h w']) -> Float[torch.Tensor, 'b c f h w']:
    latents_mean, latents_std, scaling_factor = vae.latents_mean, vae.latents_std, vae.config['scaling_factor']
    latents_mean, latents_std = latents_mean.view(1, -1, 1, 1, 1), latents_std.view(1, -1, 1, 1, 1)
    latents = rearrange(pixels, 'b f c h w -> (b f) c 1 h w').to(vae.dtype)
    latents = cast(AutoencoderKLOutput, vae.encode(latents*2-1, return_dict=True)).latent_dist.sample()
    latents = rearrange(latents, '(b f) c n h w -> b c (f n) h w', b=pixels.size(0))
    return (latents - latents_mean) / latents_std * scaling_factor

class SkipLayerStrategy(Enum):
    AttentionSkip = auto()
    AttentionValues = auto()
    Residual = auto()
    TransformerBlock = auto() # Skip the whole TransformerBlock

class LTXVideoAttnProcessorWithSkipLayerMask(LTXVideoAttnProcessor):
    def __call__(
        self,
        attn: "LTXAttention",
        hidden_states: Float[torch.Tensor, 'batch_size num_tokens inner_dim'], # latents
        encoder_hidden_states: Optional[Float[torch.Tensor, 'batch_size max_length=256 inner_dim']] = None, # text encoder outputs
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        skip_layer_mask: Optional[Float[torch.Tensor, 'batch_size']] = None, # to be implemented
        skip_layer_strategy: Optional[SkipLayerStrategy] = None, # to be implemented
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states) # Float[torch.Tensor, 'batch_size num_tokens inner_dim']
        key = attn.to_k(encoder_hidden_states) # Float[torch.Tensor, 'batch_size num_tokens inner_dim']
        value = attn.to_v(encoder_hidden_states) # Float[torch.Tensor, 'batch_size num_tokens inner_dim']

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1)) # Float[torch.Tensor, 'batch_size num_tokens num_heads head_dim']
        key = key.unflatten(2, (attn.heads, -1)) # Float[torch.Tensor, 'batch_size num_tokens num_heads head_dim']
        value = value.unflatten(2, (attn.heads, -1)) # Float[torch.Tensor, 'batch_size num_tokens num_heads head_dim']

        attn_out = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        ) # Float[torch.Tensor, 'batch_size num_tokens num_heads head_dim']
        attn_out = attn_out.flatten(2, 3) # Float[torch.Tensor, 'batch_size num_tokens num_heads*head_dim']
        attn_out = attn_out.to(query.dtype)

        if skip_layer_mask is not None:
            skip_layer_mask = skip_layer_mask.view(-1, 1, 1) # b 1 1
            if skip_layer_strategy == SkipLayerStrategy.AttentionSkip:
                hidden_states = attn_out * skip_layer_mask + hidden_states * (1.0 - skip_layer_mask)
            elif skip_layer_strategy == SkipLayerStrategy.AttentionValues:
                hidden_states = attn_out * skip_layer_mask + rearrange(value, 'b s h c -> b s (h c)') * (1.0 - skip_layer_mask)
            else:
                hidden_states = attn_out
        else:
            hidden_states = attn_out

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states) # Float[torch.Tensor, 'batch_size num_tokens inner_dim']
        return hidden_states

@maybe_allow_in_graph
class LTXVideoTransformerBlock(_LTXVideoTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super(_LTXVideoTransformerBlock, self).__init__()
        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = LTXAttention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=LTXVideoAttnProcessorWithSkipLayerMask()
        )
        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = LTXAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
        )
        self.ff = FeedForward(dim, activation_fn=activation_fn)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, 'batch_size num_tokens inner_dim'], # latents
        encoder_hidden_states: Float[torch.Tensor, 'batch_size max_length=256 inner_dim'], # text encoder outputs
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        skip_layer_mask: Optional[Float[torch.Tensor, 'batch_size']] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
    ) -> torch.Tensor:
        if (skip_layer_mask is not None) and (skip_layer_strategy == SkipLayerStrategy.TransformerBlock):
            original_hidden_states = hidden_states

        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb,
            skip_layer_mask=skip_layer_mask,
            skip_layer_strategy=skip_layer_strategy,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        if (skip_layer_mask is not None) and (skip_layer_strategy == SkipLayerStrategy.TransformerBlock):
            # Skip whole transformer block for some samples in the batch
            skip_layer_mask = skip_layer_mask.view(-1, 1, 1) # b 1 1
            hidden_states = hidden_states * skip_layer_mask + original_hidden_states * (1.0 - skip_layer_mask)

        return hidden_states

class LTXVideoTransformer3DModel(_LTXVideoTransformer3DModel):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 128, out_channels: Optional[int] = 128,
        patch_size: int = 1, patch_size_t: int = 1,
        num_attention_heads: int = 32, attention_head_dim: int = 64, cross_attention_dim: int = 2048,
        num_layers: int = 28, activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads", norm_elementwise_affine: bool = False, norm_eps: float = 1e-6,
        caption_channels: int = 4096, attention_bias: bool = True, attention_out_bias: bool = True,

        latents_add_in_channels: Optional[int] = None
    ) -> None:
        # super().__init__(
        #     in_channels=in_channels, out_channels=out_channels,
        #     patch_size=patch_size, patch_size_t=patch_size_t,
        #     num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim, cross_attention_dim=cross_attention_dim,
        #     num_layers=num_layers, activation_fn=activation_fn, qk_norm=qk_norm, norm_elementwise_affine=norm_elementwise_affine, norm_eps=norm_eps,
        #     caption_channels=caption_channels, attention_bias=attention_bias, attention_out_bias=attention_out_bias,
        # )
        super(_LTXVideoTransformer3DModel, self).__init__()

        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.time_embed = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.rope = LTXVideoRotaryPosEmbed(
            dim=inner_dim,
            base_num_frames=20,
            base_height=2048,
            base_width=2048,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            theta=10000.0,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LTXVideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.gradient_checkpointing = False

        # We can add custom projection modules here.
        if latents_add_in_channels is not None:
            # TODO: apply
            # additional (b fhw latents_add_in_channels) shaped tensor
            self.latents_add_proj_in = nn.Linear(latents_add_in_channels, inner_dim)
            with torch.no_grad():
                self.latents_add_proj_in.weight.data.zero_()
                self.latents_add_proj_in.bias.data.zero_()
        else:
            self.latents_add_proj_in = None

    @overload
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Union[Tuple[float, float, float], torch.Tensor]] = None,
        video_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: Literal[True] = True,
        **kwargs,
    ) -> Transformer2DModelOutput: ...

    @overload
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Union[Tuple[float, float, float], torch.Tensor]] = None,
        video_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: Literal[False] = False,
        **kwargs,
    ) -> Tuple[Float[torch.Tensor, 'b fhw c']]: ...

    def mocking_inputs(self, batch_size=3, num_tokens=20*30):
        return {
            'hidden_states': torch.randn((batch_size, num_tokens, self.config['in_channels'])),
            'encoder_hidden_states': torch.randn((batch_size, 256, self.config['caption_channels'])),
            'timestep': torch.randint(0, 1000, (batch_size, num_tokens), dtype=torch.long),
            'encoder_attention_mask': torch.ones((batch_size, 256), dtype=torch.bool),
            'rope_interpolation_scale': (1.0, 1.0, 1.0),
            'video_coords': torch.rand((batch_size, 3, num_tokens))
        }

    def forward(
        self,
        hidden_states: Float[torch.Tensor, 'batch_size num_tokens in_channels'], # latents
        encoder_hidden_states: Float[torch.Tensor, 'batch_size max_length=256 caption_channels'], # text encoder outputs
        timestep: Shaped[torch.Tensor, 'batch_size num_tokens'], # 0~1000, long or float, may vary across tokens because some token are condition or reference
        encoder_attention_mask: Bool[torch.Tensor, 'batch_size max_length=256'],
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Union[Tuple[float, float, float], torch.Tensor]] = None,
        video_coords: Optional[Float[torch.Tensor, 'batch_size 3 num_tokens']] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        verbose: bool = False,

        # For Spatial-Temporal Guidance (STG), skip_layer_mask[<layer_idx>][<batch_idx>]=0 means should skip the computation of this layer.
        skip_layer_mask: Optional[Float[torch.Tensor, 'num_layers batch_size']] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = SkipLayerStrategy.AttentionValues,

        # For easy conditioning. TODO: add to latents
        latents_add: Optional[Float[torch.Tensor, 'batch_size num_tokens latents_add_in_channels']] = None,
        **kwargs,
    ) -> Union[Transformer2DModelOutput, Tuple[Float[torch.Tensor, 'batch_size num_tokens out_channels']]]:

        attention_kwargs = attention_kwargs.copy() if attention_kwargs else None
        lora_scale = attention_kwargs.pop("scale", 1.0) if attention_kwargs else 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        assert (num_frames is not None and height is not None and width is not None) or (video_coords is not None)
        image_rotary_emb: PrecomputedRotaryEmb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale, video_coords)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            # Float[torch.Tensor, 'batch_size 1 max_length=256']

        batch_size = hidden_states.size(0)
        # print(f"{hidden_states.device=}")
        hidden_states = self.proj_in(hidden_states)
        # Float[torch.Tensor, 'batch_size num_tokens inner_dim']

        temb, embedded_timestep = self.time_embed(timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype)
        temb = rearrange(temb, '(b n) c -> b n c', b=batch_size)
        embedded_timestep = rearrange(embedded_timestep, '(b n) c -> b n c', b=batch_size)

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        # encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, encoder_hidden_states.size(-1))
        # Float[torch.Tensor, 'batch_size num_tokens inner_dim']

        for i, block in enumerate(self.transformer_blocks):
            # if i == len(self.transformer_blocks) // 2:
            if i == 0:
                # if self.latents_add_proj_in is not None:
                if latents_add is not None:
                    assert latents_add is not None, "latents_add cannot be None when latents_add_proj_in is set"
                    assert hasattr(self, 'latents_add_proj_in'), "latents_add_in_channels was not set during initialization"
                    assert self.latents_add_proj_in is not None, "latents_add_proj_in was not properly initialized"
                    hidden_states = hidden_states + self.latents_add_proj_in(latents_add)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                assert self._gradient_checkpointing_func is not None
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    encoder_attention_mask,
                    skip_layer_mask[i] if skip_layer_mask else None,
                    skip_layer_strategy,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    skip_layer_mask=skip_layer_mask[i] if skip_layer_mask is not None else None,
                    skip_layer_strategy=skip_layer_strategy,
                )

        # [None, None, 2, inner_dim] + [batch_size num_tokens None inner_dim]
        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        # log(f"{describe(shift)=} {describe(scale)=} {describe(self.scale_shift_table)=}")
        # describe(shift)='Tensor([batch_size, num_tokens, inner_dim])'
        # describe(scale)='Tensor([batch_size, num_tokens, inner_dim])'
        # describe(self.scale_shift_table)='Parameter([2, inner_dim])'

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output: Float[torch.Tensor, 'batch_size num_tokens out_channels'] = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        if return_dict:
            return Transformer2DModelOutput(sample=output)
        else:
            return (output,)

    @overload
    def create_skip_layer_mask(self, batch_size: int, num_conds: int, ptb_index: int, skip_block_list: None = None) -> None: ...

    @overload
    def create_skip_layer_mask(self, batch_size: int, num_conds: int, ptb_index: int, skip_block_list: List[int]) -> Float[torch.Tensor, 'num_layers num_conds*batch_size']: ...

    def create_skip_layer_mask(self, batch_size: int, num_conds: int, ptb_index: int, skip_block_list: Optional[List[int]] = None) -> Optional[Float[torch.Tensor, 'num_layers num_conds*batch_size']]:
        if skip_block_list is None or len(skip_block_list) == 0:
            return None
        num_layers = len(self.transformer_blocks)
        mask = torch.ones((num_layers, num_conds * batch_size), device=self.device, dtype=self.dtype)
        for block_idx in skip_block_list:
            mask[block_idx, ptb_index::num_conds] = 0
        return mask

# 像素到 DiT 输入的过程:
# pixels --(vae,normalize)-> Latents --(patchify)-> PatchifiedLatents --(flatten)-> TokenizedLatents -> LTXVideoTransformer3DModel(hidden_states)
# 但每个像素用于 RoPE 编码的坐标有更复杂的过程

# VAE 输出的 latents
Latents = Float[torch.Tensor, 'b c f h w']
LatentMask = Float[torch.Tensor, 'b 1 f h w'] # =1 means is conditioning; =0 means not conditioning
LatentConditioning = Tuple[Latents, LatentMask]

# latents 在各自张量中的序号
LatentIndices = Integer[torch.Tensor, 'b f h w c=3']
def latent_indices(latents: Latents) -> LatentIndices:
    """每个 latent 的 (f, h, w) 序号, 以 8*32*32 个像素为单位"""
    b, c, f, h, w = latents.shape
    with latents.device:
        latent_sample_coords = torch.meshgrid(torch.arange(0, f), torch.arange(0, h), torch.arange(0, w), indexing='ij')
        latent_sample_coords = torch.stack(latent_sample_coords, dim=-1)
        return latent_sample_coords.unsqueeze(0).repeat(b, 1, 1, 1, 1)

# 经过 Patchify 之后的 latents
PatchifiedLatents = Float[torch.Tensor, 'b fp hp wp c']
PatchIndices = Integer[torch.Tensor, 'b f h w c=3']
def patch_indices(patches: PatchifiedLatents, patch_size=(1, 1, 1)) -> PatchIndices:
    """每个 patch 最左上角的 latent 的 (f h w) 序号, 单位是 latent 而非 patch, 即与 LatentIndices 是同一坐标系"""
    b, fp, hp, wp, c = patches.shape
    pf, ph, pw = patch_size
    with patches.device:
        patches_sample_coords = torch.meshgrid(torch.arange(0, fp*pf, pf), torch.arange(0, hp*ph, ph), torch.arange(0, wp*pw, pw), indexing='ij')
        patches_sample_coords = torch.stack(patches_sample_coords, dim=-1)
        return patches_sample_coords.unsqueeze(0).repeat(b, 1, 1, 1, 1)

def indices_to_pixel_coords(indices: Union[LatentIndices, PatchIndices], temporal_compress: int = 8, spatial_compress: int = 32, frame_offset: int = 0) -> Integer[torch.Tensor, 'b f h w c=3']:
    with indices.device:
        # 0,1,2,3 -> 0,8,16,24
        indices = indices * torch.tensor([temporal_compress, spatial_compress, spatial_compress])
        # 0,1,2,3 -> 0,8,16,24 -> 0,1,9,17
        indices[..., 0] = (indices[..., 0] + 1 - temporal_compress).clamp(min=0)
        # 0,1,2,3 -> 0,8,16,24 -> 0+offset,1+offset,9+offset,17+offset
        indices[..., 0] += frame_offset
    return indices

TokenizedVideoCoords = Float[torch.Tensor, 'b c=3 f*h*w']
def pixel_to_video_coords(pixel_coords: Integer[torch.Tensor, 'b f h w c=3'], frame_rate: float = 25) -> TokenizedVideoCoords:
    video_coords = rearrange(pixel_coords, 'b f h w c -> b c (f h w)').float()
    video_coords[:, 0] /= frame_rate
    return video_coords

# 展平为 token sequence 的 latents
TokenizedLatents = Float[torch.Tensor, 'batch_size num_tokens in_channels']
TokenizedLatentMask = Float[torch.Tensor, 'batch_size num_tokens in_channels']
TokenizedLatentConditioning = Tuple[TokenizedLatents, TokenizedLatentMask]

@overload
def patchify(latents: None, patch_size=(1, 1, 1)) -> None: ...

@overload
def patchify(latents: Latents, patch_size=(1, 1, 1)) -> PatchifiedLatents: ...

def patchify(latents: Optional[Latents], patch_size=(1, 1, 1)) -> Optional[PatchifiedLatents]:
    if latents is None:
        return None
    pf, ph, pw = patch_size
    return rearrange(latents, 'b c (fp pf) (hp ph) (wp pw) -> b fp hp wp (c pf ph pw)', pf=pf, ph=ph, pw=pw)

@overload
def tokenize(patches: None) -> None: ...

@overload
def tokenize(patches: PatchifiedLatents) -> TokenizedLatents: ...

def tokenize(patches: Optional[PatchifiedLatents]) -> Optional[TokenizedLatents]:
    if patches is None:
        return None
    return rearrange(patches, 'b fp hp wp c -> b (fp hp wp) c')

TextEmbedsTensor = Float[torch.Tensor, 'b s=256 c=1024']
TextEmbedAttnMaskTensor = Bool[torch.Tensor, 'b s=256']
TextEmbeds = TypedDict('TextEmbeds', {"prompt_embeds": TextEmbedsTensor, "prompt_attention_mask": TextEmbedAttnMaskTensor})
Prompts = Union[TextEmbeds, List[str]]

# t b f h w -> float
GuidanceScale = Union[torch.Tensor, float] # something that can be multiplied with latent tensor
DynamicGuidanceScale = Callable[[Union[float, torch.Tensor]], GuidanceScale]
GuidanceScaleSource = Union[DynamicGuidanceScale, GuidanceScale]

class BaseLTXVPipeline:
    def set_progress_bar_config(self, disable):
        pass

    @classmethod
    def from_pipeline(cls, pipeline) -> BaseLTXVPipeline:
        return cls(
            scheduler=pipeline.scheduler,
            vae=pipeline.vae,
            transformer=pipeline.transformer,
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
        )

    @torch.no_grad()
    def encode_prompts(self, prompts: Prompts, max_sequence_length: int = 256, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> TextEmbeds:
        return encode_prompts(self.tokenizer, self.text_encoder, prompts, max_sequence_length=max_sequence_length, device=device, dtype=dtype)

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTXVideo,
        transformer: LTXVideoTransformer3DModel,
        tokenizer: Optional[Union[T5TokenizerFast, T5Tokenizer]] = None,
        text_encoder: Optional[T5EncoderModel] = None,
    ):
        self.scheduler = scheduler
        self.vae = vae
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def process_prompts(self, prompts: Prompts, negative_prompts: Prompts, device: torch.device, dtype: torch.dtype):
        text_embeds = self.encode_prompts(prompts, device=device, dtype=dtype)
        prompt_embeds, prompt_attention_mask = text_embeds["prompt_embeds"], text_embeds["prompt_attention_mask"]
        negative_text_embeds = self.encode_prompts(negative_prompts, device=device, dtype=dtype)
        negative_prompt_embeds, negative_prompt_attention_mask = negative_text_embeds["prompt_embeds"], negative_text_embeds["prompt_attention_mask"]

        batch_size = max(prompt_embeds.size(0), negative_prompt_embeds.size(0))
        if prompt_embeds.size(0) != batch_size:
            assert prompt_embeds.size(0) == 1
            prompt_embeds, prompt_attention_mask = prompt_embeds.repeat(batch_size, 1, 1), prompt_attention_mask.repeat(batch_size, 1)
        if negative_prompt_embeds.size(0) != batch_size:
            assert negative_prompt_embeds.size(0) == 1
            negative_prompt_embeds, negative_prompt_attention_mask = negative_prompt_embeds.repeat(batch_size, 1, 1), negative_prompt_attention_mask.repeat(batch_size, 1)
        return batch_size, prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def retrieve_guidance_scale(self, source: GuidanceScaleSource, timestep: Union[float, Float[torch.Tensor, '']], sample: TokenizedLatents) -> GuidanceScale:
        if callable(source):
            return self.retrieve_guidance_scale(source(timestep), timestep, sample)
        elif isinstance(source, (float, int)):
            return source # return torch.full_like(sample, fill_value=source)
        elif isinstance(source, torch.Tensor):
            return source
        else:
            raise ValueError(f"Unsupported guidance scale source: {source}")

    @torch.no_grad()
    def latents_sample_loop(
        self,

        latents: TokenizedLatents,
        latent_coords: Float[torch.Tensor, 'b 3 fp*hp*wp'], # temporal/spatial coords for each token

        prompt_embeds: TextEmbedsTensor,
        prompt_attention_mask: TextEmbedAttnMaskTensor,
        negative_prompt_embeds: TextEmbedsTensor,
        negative_prompt_attention_mask: TextEmbedAttnMaskTensor,

        conditioning: Optional[TokenizedLatentConditioning] = None, # 0~1, 1 means is conditioning, shape same as latents

        num_inference_steps: int = 50,
        cfg_source: GuidanceScaleSource = 3,
        stg_source: GuidanceScaleSource = 1,
        rescaling_source: GuidanceScaleSource = 0.7,
        cfg_star_rescale: bool = False, # only =True for 13B model.
        image_cond_noise_scale: float = 0.15,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,

        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,

        **extras
    ):
        device, dtype = (device or self.transformer.device), (dtype or self.transformer.dtype)

        # timesteps
        sigmas = linear_quadratic_schedule(num_inference_steps)
        self.scheduler.set_timesteps(timesteps=(sigmas * 1000).tolist(), device=device)
        timesteps = cast(torch.Tensor, self.scheduler.timesteps).to(device=device, dtype=dtype)
        num_inference_steps = len(timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        timesteps = timesteps[skip_initial_inference_steps:(-skip_final_inference_steps or None)]
        # assert (skip_initial_inference_steps == 0) or (init_latents is not None)

        batch_size, num_tokens, latent_c = latents.shape

        for i, t in enumerate(timesteps):
            cfg_scale = self.retrieve_guidance_scale(cfg_source, t, latents)
            stg_scale = self.retrieve_guidance_scale(stg_source, t, latents)
            rescaling = self.retrieve_guidance_scale(rescaling_source, t, latents)
            do_classifier_free_guidance = isinstance(cfg_scale, torch.Tensor) or (cfg_scale > 1.0)
            do_spatio_temporal_guidance = isinstance(stg_scale, torch.Tensor) or (stg_scale > 0.0)
            do_rescaling = isinstance(rescaling, torch.Tensor) or (rescaling != 1.0)
            num_cond = 1 + int(do_classifier_free_guidance) + int(do_spatio_temporal_guidance)
            if do_classifier_free_guidance and do_spatio_temporal_guidance:
                indices = [0, 1, 2]
            elif do_classifier_free_guidance:
                indices = [0, 1]
            elif do_spatio_temporal_guidance:
                indices = [1, 2]
            else:
                indices = [1]

            timestep = t.expand(batch_size)[..., None] # () -> b 1

            if conditioning is not None:
                condition_latents, condition_mask = conditioning
                condition_noise_scale = image_cond_noise_scale * (t / 1000) ** 2 # b 1

                # noisy_condition_latents = condition_latents + randn_like(condition_latents, generator=generator) * condition_noise_scale[..., None]
                noisy_condition_latents = torch.lerp(condition_latents, randn_like(condition_latents, generator=generator), condition_noise_scale[..., None])

                input_latents = torch.lerp(latents, noisy_condition_latents, condition_mask)
                timestep = torch.min(timestep[..., None], 1000. * (1 - condition_mask)).mean(dim=-1) # torch.min(b n -> b n 1, b n c) -> b n c -> b n
            else:
                input_latents = latents

            hidden_states_inputs = torch.cat([input_latents]*num_cond)

            timestep_inputs = torch.cat([timestep]*num_cond)
            video_coords_inputs = torch.cat([latent_coords]*num_cond)

            full_encoder_hidden_states_inputs = torch.stack([negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0) # g b ...
            full_encoder_attention_mask_inputs = torch.stack([negative_prompt_attention_mask, prompt_attention_mask, prompt_attention_mask], dim=0) # g b ...
            encoder_hidden_states_inputs = rearrange(full_encoder_hidden_states_inputs[indices], 'g b ... -> (g b) ...')
            encoder_attention_mask_inputs = rearrange(full_encoder_attention_mask_inputs[indices], 'g b ... -> (g b) ...')

            noise_pred = cast(TokenizedLatents, self.transformer(
                hidden_states=hidden_states_inputs,
                encoder_hidden_states=encoder_hidden_states_inputs,
                encoder_attention_mask=encoder_attention_mask_inputs,
                video_coords=video_coords_inputs,
                timestep=timestep_inputs,

                skip_layer_mask=self.transformer.create_skip_layer_mask(batch_size, num_cond, num_cond-1, [19]) if do_spatio_temporal_guidance else None,
                skip_layer_strategy=SkipLayerStrategy.AttentionValues if do_spatio_temporal_guidance else None,
                return_dict=True,
                **extras
            ).sample)
            noise_pred_chunks = noise_pred.chunk(num_cond)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred_chunks[:2]
                if cfg_star_rescale:
                    # Rescales the unconditional noise prediction using the projection of the conditional prediction onto it:
                    # α = (⟨ε_text, ε_uncond⟩ / ||ε_uncond||²), then ε_uncond ← α * ε_uncond
                    # where ε_text is the conditional noise prediction and ε_uncond is the unconditional one.
                    positive_flat = noise_pred_cond.view(batch_size, -1)
                    negative_flat = noise_pred_uncond.view(batch_size, -1)
                    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                    squared_norm = (torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8)
                    alpha = dot_product / squared_norm
                    noise_pred_uncond = alpha * noise_pred_uncond
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            elif do_spatio_temporal_guidance:
                noise_pred_cond, noise_pred_cond_perturb = noise_pred_chunks[-2:]
                noise_pred = noise_pred_cond

            if do_spatio_temporal_guidance:
                noise_pred_cond, noise_pred_cond_perturb = noise_pred_chunks[-2:]
                noise_pred = noise_pred + stg_scale * (noise_pred_cond - noise_pred_cond_perturb)
                if do_rescaling and do_spatio_temporal_guidance:
                    noise_pred_cond_std = noise_pred_cond.view(batch_size, -1).std(dim=1, keepdim=True)
                    noise_pred_std = noise_pred.view(batch_size, -1).std(dim=1, keepdim=True)

                    factor = noise_pred_cond_std / noise_pred_std
                    factor = rescaling * factor + (1 - rescaling)

                    noise_pred = noise_pred * factor.view(batch_size, 1, 1)

            denoised_latents = cast(FlowMatchEulerDiscreteSchedulerOutput, self.scheduler.step(
                -noise_pred, timestep=t, sample=input_latents, per_token_timesteps=timestep, return_dict=True # type: ignore
                # noise_pred, timestep=t, sample=input_latents, per_token_timesteps=None, return_dict=True # type: ignore
            )).prev_sample

            if conditioning is None:
                latents = denoised_latents
            else:
                condition_latents, condition_mask = conditioning
                latents_to_denoise_mask = timestep[..., None] / 1000 < (1.0 - condition_mask) + 1e-3
                latents = torch.where(latents_to_denoise_mask, denoised_latents, condition_latents)
        return latents
        # untokenize & unpatchify
        # return rearrange(latents, 'b (fp hp wp) (c pf ph pw) -> b c (fp pf) (hp ph) (wp pw)', fp=latent_f, hp=latent_h, wp=latent_w, pf=1, ph=1, pw=1)

    @torch.no_grad()
    def sample_latents(
        self,
        prompts: Prompts,
        negative_prompts: Prompts = ['worst quality, inconsistent motion, blurry, jittery, distorted'],

        num_frames: int = 81, height: int = 768, width: int = 1024, frame_rate: int = 25,

        conditioning: Optional[LatentConditioning] = None,
        image_cond_noise_scale: float = 0.15,

        # init_latents 必须是经过 normalize 的，由调用者保证
        init_latents: Optional[Latents] = None,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,

        manual_token_coords: Optional[Float[torch.Tensor, 'b c fhw']] = None,

        num_inference_steps: int = 50,
        cfg_source: GuidanceScaleSource = 3,
        stg_source: GuidanceScaleSource = 1,
        rescaling_source: GuidanceScaleSource = 0.7,
        cfg_star_rescale: bool = False, # only =True for 13B model.

        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Latents:
        device, dtype = (device or self.transformer.device), (dtype or self.transformer.dtype)

        # timesteps
        sigmas = linear_quadratic_schedule(num_inference_steps)
        self.scheduler.set_timesteps(timesteps=(sigmas * 1000).tolist(), device=device)
        timesteps = cast(torch.Tensor, self.scheduler.timesteps).to(device=device, dtype=dtype)
        num_inference_steps = len(timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        timesteps = timesteps[skip_initial_inference_steps:(-skip_final_inference_steps or None)]
        assert (skip_initial_inference_steps == 0) or (init_latents is not None)
        # print(f"{timesteps=}") # check passed

        # prompts
        batch_size, prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = self.process_prompts(prompts, negative_prompts, device, dtype)
        # print(f"{prompt_attention_mask.float().mean()}") # check passed

        # latent dimensions
        t_compress, s_compress = self.vae.temporal_compression_ratio, self.vae.spatial_compression_ratio
        assert (num_frames - 1) % t_compress == 0
        assert height % s_compress == 0
        assert width % s_compress == 0

        # i2v / v2v initial latents
        if init_latents is not None:
            latent_b, latent_c, latent_f, latent_h, latent_w = init_latents.shape
            if latent_b != batch_size:
                assert latent_b == 1
                latent_b = batch_size
                init_latents = init_latents.expand(batch_size, -1, -1, -1, -1)
            num_frames = (latent_f - 1) * t_compress + 1
            height = latent_h * s_compress
            width = latent_w * s_compress

            latent_shape = init_latents.shape
            latents = init_latents.to(device=device, dtype=dtype)
        else:
            latent_b, latent_c, latent_f, latent_h, latent_w = batch_size, self.transformer.config.get('in_channels', 128), (num_frames - 1) // t_compress + 1, height // s_compress, width // s_compress
            latent_shape = (latent_b, latent_c, latent_f, latent_h, latent_w)
            latents = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)

        patches = patchify(latents)
        video_indices = patch_indices(patches) # _prepare_video_ids
        video_coords = indices_to_pixel_coords(video_indices) # _scale_video_ids

        # print(f"{rearrange(video_coords, 'b f h w c -> b c (f h w)')=}") # check passed

        if conditioning is not None:
            condition_latents, condition_mask = conditioning
            condition_mask = condition_mask.expand(-1, latents.size(1), -1, -1, -1) # b 1 f h w -> b 128 f h w
            condition_patches = patchify(condition_latents.to(device=device, dtype=dtype))
            condition_mask_patches = patchify(condition_mask.to(device=device, dtype=dtype))
            condition_tokens = tokenize(condition_patches)
            condition_mask_tokens = tokenize(condition_mask_patches)
        else:
            # condition_tokens = torch.empty_like(tokens)
            # condition_mask_tokens = torch.empty_like(tokens)

            # we will not try to access these two objects when conditioning is None; but type checher must be fooled.
            condition_tokens = cast(torch.Tensor, None)
            condition_mask_tokens = cast(torch.Tensor, None)

        ####### below this line, we only operate on tokenized latents (b fhw c) #######

        # denoise on tokens
        tokens = tokenize(patches)

        if conditioning is None:
            init_tokens = tokens.clone() # Used for image_cond_noise_update
        else:
            init_tokens = torch.lerp(tokens, condition_tokens, condition_mask_tokens)

        if manual_token_coords is None:
            token_coords = rearrange(video_coords, 'b f h w c -> b c (f h w)').float()
            token_coords[:, 0] = token_coords[:, 0] / frame_rate
        else:
            token_coords = manual_token_coords
        # NOTE: get_rope_scale_factors = indices_to_pixel_coords & ([:, 0] / frame_rate)

        for i, t in enumerate(timesteps):
            cfg_scale = self.retrieve_guidance_scale(cfg_source, t, tokens)
            stg_scale = self.retrieve_guidance_scale(stg_source, t, tokens)
            rescaling = self.retrieve_guidance_scale(rescaling_source, t, tokens)
            do_classifier_free_guidance = isinstance(cfg_scale, torch.Tensor) or (cfg_scale > 1.0)
            do_spatio_temporal_guidance = isinstance(stg_scale, torch.Tensor) or (stg_scale > 0.0)
            do_rescaling = isinstance(rescaling, torch.Tensor) or (rescaling != 1.0)

            # 0: unconditional; 1: conditional text; 2: conditional text + skip layer
            cfg_indices = [0, 1] if do_classifier_free_guidance else []
            stg_indices = [1, 2] if do_spatio_temporal_guidance else []
            indices = sorted(set(cfg_indices + stg_indices))
            num_cond = len(indices)

            # () -> b 1
            timestep = t.expand(batch_size)[..., None] # b 1
            # TODO: apply condition mask here; b 1 -> b fhw

            if conditioning is not None:
                assert tokens.shape == condition_tokens.shape, f"{tokens.shape=} {condition_tokens.shape=} must match"
                noise = torch.randn(condition_tokens.shape, generator=generator, device=device)
                need_to_noise = condition_mask_tokens > 1.0 - 1e-3
                noised_condition_tokens = condition_tokens + image_cond_noise_scale * noise * ((timestep / 1000)**2)[..., None]

                input_tokens = torch.where(need_to_noise, noised_condition_tokens, tokens)

                timestep = torch.min(timestep, (1 - condition_mask_tokens.mean(dim=-1)) * 1000)
            else:
                input_tokens = tokens

            hidden_states_inputs = torch.cat([input_tokens]*num_cond)

            timestep_inputs = torch.cat([timestep]*num_cond)
            video_coords_inputs = torch.cat([token_coords]*num_cond)

            full_encoder_hidden_states_inputs = torch.stack([negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0) # g b ...
            full_encoder_attention_mask_inputs = torch.stack([negative_prompt_attention_mask, prompt_attention_mask, prompt_attention_mask], dim=0) # g b ...
            encoder_hidden_states_inputs = rearrange(full_encoder_hidden_states_inputs[indices], 'g b ... -> (g b) ...')
            encoder_attention_mask_inputs = rearrange(full_encoder_attention_mask_inputs[indices], 'g b ... -> (g b) ...')

            # from gshub.utils import describe
            # print(f"{hidden_states_inputs.std(dim=1)=}")
            # print(f"{describe(encoder_hidden_states_inputs)=}")
            # print(f"{encoder_hidden_states_inputs=}")
            # print(f"{timestep_inputs=}")
            # print(f"{video_coords_inputs.std(dim=1)=}")

            noise_pred = cast(TokenizedLatents, self.transformer(
                hidden_states=hidden_states_inputs,
                encoder_hidden_states=encoder_hidden_states_inputs,
                encoder_attention_mask=encoder_attention_mask_inputs,
                video_coords=video_coords_inputs,
                timestep=timestep_inputs,

                skip_layer_mask=self.transformer.create_skip_layer_mask(batch_size, num_cond, num_cond-1, [19]) if do_spatio_temporal_guidance else None,
                skip_layer_strategy=SkipLayerStrategy.AttentionValues if do_spatio_temporal_guidance else None,
                return_dict=True,
            ).sample)

            noise_pred_chunks = noise_pred.chunk(num_cond)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred_chunks[:2]
                if cfg_star_rescale:
                    # Rescales the unconditional noise prediction using the projection of the conditional prediction onto it:
                    # α = (⟨ε_text, ε_uncond⟩ / ||ε_uncond||²), then ε_uncond ← α * ε_uncond
                    # where ε_text is the conditional noise prediction and ε_uncond is the unconditional one.
                    positive_flat = noise_pred_cond.view(batch_size, -1)
                    negative_flat = noise_pred_uncond.view(batch_size, -1)
                    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                    squared_norm = (torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8)
                    alpha = dot_product / squared_norm
                    noise_pred_uncond = alpha * noise_pred_uncond
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            elif do_spatio_temporal_guidance:
                noise_pred_cond, noise_pred_cond_perturb = noise_pred_chunks[-2:]
                noise_pred = noise_pred_cond

            if do_spatio_temporal_guidance:
                noise_pred_cond, noise_pred_cond_perturb = noise_pred_chunks[-2:]
                noise_pred = noise_pred + stg_scale * (noise_pred_cond - noise_pred_cond_perturb)
                if do_rescaling and do_spatio_temporal_guidance:
                    noise_pred_cond_std = noise_pred_cond.view(batch_size, -1).std(dim=1, keepdim=True)
                    noise_pred_std = noise_pred.view(batch_size, -1).std(dim=1, keepdim=True)

                    factor = noise_pred_cond_std / noise_pred_std
                    factor = rescaling * factor + (1 - rescaling)

                    noise_pred = noise_pred * factor.view(batch_size, 1, 1)

            denoised_tokens = cast(FlowMatchEulerDiscreteSchedulerOutput, self.scheduler.step(
                -noise_pred, timestep=t, sample=input_tokens, per_token_timesteps=timestep, return_dict=True # type: ignore
            )).prev_sample

            if conditioning is None:
                tokens = denoised_tokens
            else:
                tokens_to_denoise_mask = timestep[..., None] / 1000 < (1.0 - condition_mask_tokens) + 1e-3
                tokens = torch.where(tokens_to_denoise_mask, denoised_tokens, condition_tokens)
        # untokenize & unpatchify
        return rearrange(tokens, 'b (fp hp wp) (c pf ph pw) -> b c (fp pf) (hp ph) (wp pw)', fp=latent_f, hp=latent_h, wp=latent_w, pf=1, ph=1, pw=1)

    @torch.no_grad()
    def sample_videos(
        self,
        normalized_latents: Latents,
        decode_timestep: Union[float, List[float]] = 0.05,
        decode_noise_scale: Union[float, List[float]] = 0.025,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
    ):
        device, dtype = (device or self.vae.device), (dtype or self.vae.dtype)
        normalized_latents = normalized_latents.to(device, dtype)
        latents_mean, latents_std, scaling_factor = self.vae.latents_mean.to(normalized_latents), self.vae.latents_std.to(normalized_latents), self.vae.config['scaling_factor']
        latents_mean, latents_std = latents_mean.view(1, -1, 1, 1, 1), latents_std.view(1, -1, 1, 1, 1)
        latents = normalized_latents * latents_std / scaling_factor + latents_mean
        b, c, f, h, w = latents.shape

        with device:
            timestep_conditioning = self.vae.config['timestep_conditioning']
            if not timestep_conditioning:
                decode_timesteps = None
            else:
                decode_timesteps = torch.tensor([decode_timestep]*b if isinstance(decode_timestep, float) else decode_timestep).to(dtype)
                decode_noise_scales = torch.tensor([decode_noise_scale]*b if isinstance(decode_noise_scale, float) else decode_noise_scale).to(dtype)[:, None, None, None, None]
                noise = randn_like(latents, generator=generator)
                latents = torch.lerp(latents, noise, decode_noise_scales)

            # def _run_decoder(latents: torch.Tensor, vae: AutoencoderKLLTXVideo, is_video: bool, vae_per_channel_normalize=False, timestep=None) -> torch.Tensor:
            #     if isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)):
            #         *_, fl, hl, wl = latents.shape
            #         temporal_scale, spatial_scale, _ = get_vae_size_scale_factor(vae)
            #         latents = latents.to(vae.dtype)
            #         vae_decode_kwargs = {}
            #         if timestep is not None:
            #             vae_decode_kwargs["timestep"] = timestep
            #         image = vae.decode(
            #             un_normalize_latents(latents, vae, vae_per_channel_normalize),
            #             return_dict=False,
            #             target_shape=(
            #                 1,
            #                 3,
            #                 fl * temporal_scale if is_video else 1,
            #                 hl * spatial_scale,
            #                 wl * spatial_scale,
            #             ),
            #             **vae_decode_kwargs,
            #         )[0]
            #     else:
            #         image = vae.decode(un_normalize_latents(latents, vae, vae_per_channel_normalize), return_dict=False)[0]
            #     return image
            # def vae_decode(latents: torch.Tensor, vae: AutoencoderKLLTXVideo, is_video: bool = True, split_size: int = 1, vae_per_channel_normalize=False, timestep=None) -> torch.Tensor:
            #     is_video_shaped = latents.dim() == 5
            #     batch_size = latents.shape[0]
            #     # if is_video_shaped and not isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)):
            #     #     latents = rearrange(latents, "b c n h w -> (b n) c h w")
            #     images = _run_decoder(latents, vae, is_video, vae_per_channel_normalize, timestep)
            #     # if is_video_shaped and not isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)):
            #     #     images = rearrange(images, "(b n) c h w -> b c n h w", b=batch_size)
            #     return images
            # video = vae_decode(latents, self.vae, is_video=True, vae_per_channel_normalize=True, timestep=decode_timesteps)
            video = cast(DecoderOutput, self.vae.decode(latents, decode_timesteps, return_dict=True)).sample
            return video * .5 + .5

# RectifiedFlowScheduler.from_pretrained(ckpt_path)

if __name__ == "__main__":
    from gshub.utils import best_cuda_device
    from tqdm.auto import tqdm
    device = best_cuda_device()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--devices', type=str, nargs='+', default=[0], help='List of devices to use')
    args = parser.parse_args()
    devices = [torch.device(f'cuda:{i}') for i in args.devices]
    args.size = len(devices)
    # xfmr = LTXVideoTransformer3DModel()
    # mocking_inputs = xfmr.mocking_inputs()
    # with torch.no_grad():
    #     outputs = xfmr(**mocking_inputs, return_dict=True).sample

    def load_tokenizer() -> T5TokenizerFast:
        return T5TokenizerFast.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="tokenizer",
        )

    def load_text_encoder(*, load_in_8bit: bool = False) -> T5EncoderModel:
        kwargs = (
            {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
            if load_in_8bit else {"torch_dtype": torch.bfloat16}
        )
        return T5EncoderModel.from_pretrained("Lightricks/LTX-Video", subfolder="text_encoder", **kwargs) # type: ignore

    tokenizer = load_tokenizer()
    text_encoder = load_text_encoder().to(device) # type: ignore
    captioner = create_captioner(CaptionerType.QWEN_25_VL, device=device)

    for scene in tqdm(sorted(list(Path('../gshub/output/synthesized_dl3dv10k/').iterdir()))[args.rank::len(devices)]):
        if (scene / '.done').is_file():
            if (scene / '.caption.done').is_file():
                continue
            try:
                caption = captioner.caption(scene / 'ground_truth.mp4')
                text_embeds = encode_prompts(tokenizer, text_encoder, [caption], device=device, dtype=text_encoder.dtype)
                text_embeds = {
                    "prompt_embeds": text_embeds["prompt_embeds"].squeeze(0).cpu(),
                    "prompt_attention_mask": text_embeds["prompt_attention_mask"].squeeze(0).cpu(),
                }
                torch.save(text_embeds, scene / 'prompt.pt')
                with open(scene / 'caption.txt', 'w') as f:
                    f.write(caption)
                (scene / '.caption.done').touch(exist_ok=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing {scene}: {e}")