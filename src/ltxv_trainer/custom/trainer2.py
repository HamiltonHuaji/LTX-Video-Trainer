from __future__ import annotations

import math
import copy
import random
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional

from gshub.utils import describe, not_none, not_none_or_dotenv

from einops import *
from typing import *
from typing_extensions import *
from jaxtyping import Float, Shaped, Int, Integer, Bool
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder
import imageio.v3 as iio

from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Group, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from ltxv_trainer import logger
from ltxv_trainer.config import *
from ltxv_trainer.config import ConditioningConfig
from ltxv_trainer.trainer import LtxvTrainer
from ltxv_trainer.timestep_samplers import TimestepSampler
from ltxv_trainer.training_strategies import TrainingStrategy
from ltxv_trainer.quantization import quantize_model
from ltxv_trainer.model_loader import LtxvModelVersion, load_vae
from ltxv_trainer.custom.base import CustomDL3DV10KDatasetBatch, CustomDL3DV10KDataset, load_ltxv_xfmr_2b_manually

from ltxv_trainer.custom.inference import BaseLTXVPipeline, LTXVideoTransformer3DModel, Latents, LatentConditioning, latent_indices, patch_indices, indices_to_pixel_coords, patchify, tokenize, encode_as_video, encode_as_images, randn_like

from diffusers import BitsAndBytesConfig
from diffusers.models.autoencoders import AutoencoderKLLTXVideo
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput, Transformer2DModelOutput
from transformers import T5EncoderModel, T5Tokenizer, T5TokenizerFast
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerOutput

def load_scheduler() -> FlowMatchEulerDiscreteScheduler:
    # Use the latest scheduler config from LTXV_13B_097_DEV.
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        LtxvModelVersion.LTXV_13B_097_DEV.hf_repo,
        subfolder="scheduler",
    )

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
    return T5EncoderModel.from_pretrained("Lightricks/LTX-Video", subfolder="text_encoder", **kwargs)

class VideoFrame(BaseModel):
    path: str
    index: int = Field(default=0)
class Video(BaseModel):
    path: str
    start: int = 0
    end: Optional[int] = None
    step: Optional[int] = None
    fps_override: Optional[int] = None
class ValidationSample(BaseModel):
    model_config = ConfigDict(extra="allow")

    reference_video: str | Video
    condition_video: str | Video

    first_frame: Optional[str | VideoFrame] = None

    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
class CustomValidationConfig(ValidationConfig):
    model_config = ConfigDict(
        populate_by_name=True,
        extra='ignore'
    )

    default_prompt: Optional[str] = Field(
        default='Walking on the pavements of an inner road through a shopping centre. The camera translates rightward horizontally and smoothly.',
        description="Default positive prompt",
    )
    default_negative_prompt: Optional[str] = Field(
        default='worst quality, inconsistent motion, blurry, jittery, distorted',
        description="Default negative prompt",
    )

    samples: List[ValidationSample] = Field(
        default_factory=list,
        description="List of validation samples",
    )
    stg_scale: float = Field(default=1.0, description="Spatio-temporal guidance scale")
    cfg_scale: float = Field(default=3.0, description="Classifier-free guidance scale", alias='guidance_scale')
    rescaling_scale: float = Field(default=0.7, description="Rescaling scale")

    guidance_scale: Optional[float] = Field(default=None, exclude=True, deprecated=True)

class DINOv3Config(BaseModel):
    model_name: str = Field(default='dinov3_vitl16')
    repository: str | None = Field(default=None)
    checkpoint: str | None = Field(default=None)
    source: str = Field(default='local', description="One of ['local', 'github']")

    def model(self):
        from gshub.utils import not_none_or_dotenv
        from dinov3.models.vision_transformer import DinoVisionTransformer

        dinov3_source = not_none_or_dotenv(self.source, 'dinov3_source')
        dinov3_repository = not_none_or_dotenv(self.repository, 'dinov3_repository')
        dinov3_checkpoint = not_none_or_dotenv(self.checkpoint, 'dinov3_vitl16_pretrain_lvd1689m')
        return cast(DinoVisionTransformer, torch.hub.load(dinov3_repository, self.model_name, source=dinov3_source, weights=dinov3_checkpoint))

    @classmethod
    def transform(cls): # pixels: Float[torch.Tensor, 'b c h w']
        return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class Pi3Config(BaseModel):
    model_name: str = Field(default='yyfz233/Pi3')
    lora_path: str | None = Field(default=None)
    lora_rank: int = 128
    lora_alpha: int = 128

    def model(self):
        from pi3.models.pi3 import Pi3
        from peft import inject_adapter_in_model, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
        model = Pi3.from_pretrained(self.model_name)

        if self.lora_path:
            lora_path = Path(self.lora_path)
            if lora_path.name.endswith('.safetensors'):
                from safetensors.torch import load_file
                lora_state_dict = load_file(lora_path)
            else:
                lora_state_dict = torch.load(lora_path, map_location='cpu')
            lora_config = LoraConfig(r=self.lora_rank, lora_alpha=self.lora_alpha, target_modules=[
                    f'encoder.blocks.{i}.attn.qkv' for i in range(24)
                ] + [
                    f'encoder.blocks.{i}.attn.proj' for i in range(24)
                ] + [
                    f'decoder.{i}.attn.qkv' for i in range(36)
                ] + [
                    f'decoder.{i}.attn.proj' for i in range(36)
                ]
            )
            if 'state_dict' in lora_state_dict:
                # which means the lora_path points to a pytorch lightning full checkpoint
                model = inject_adapter_in_model(lora_config, model)
                model.load_state_dict(lora_state_dict['state_dict'], strict=False) # type: ignore
            else:
                # the lora_path points to a lora-only weight.
                model = inject_adapter_in_model(lora_config, model, state_dict=lora_state_dict)
                set_peft_model_state_dict(model, lora_state_dict)
        return cast(Pi3, model)

    @classmethod
    def transform(cls): # pixels: Float[torch.Tensor, 'b n c h w']
        return torchvision.transforms.Normalize(mean=[0,0,0], std=[1,1,1]) # normalization is done internally in the model

class CustomTrainerConfig(LtxvTrainerConfig):
    validation: CustomValidationConfig = Field(default_factory=CustomValidationConfig)

    dinov3: DINOv3Config = Field(default_factory=DINOv3Config)
    pi3: Pi3Config = Field(default_factory=Pi3Config)

    @model_validator(mode="after")
    def validate_conditioning_compatibility(self) -> Self:
        return self

# CustomDL3DV10KDatasetBatch:
# describe(batch)={'fps': 'Tensor([1],torch.float64,cuda:0)', 'prompt_embeds': 
# 'Tensor([1, 256, 4096],torch.float16,cuda:0)', 'prompt_attention_mask': 
# 'Tensor([1, 256],torch.bool,cuda:0)', 'target_pixels': 'Tensor([1, 25, 3, 640, 
# 960],torch.float32,cuda:0)', 'condition_pixels': 'Tensor([1, 25, 3, 640, 
# 960],torch.float32,cuda:0)', 'condition_depths': 'Tensor([1, 25, 3, 640, 
# 960],torch.float32,cuda:0)', 'condition_extrinsics': 'Tensor([1, 25, 4, 
# 4],torch.float32,cuda:0)', 'condition_intrinsics': 'Tensor([1, 25, 3, 
# 3],torch.float32,cuda:0)', 'reference_pixels': 'Tensor([1, 25, 3, 640, 
# 960],torch.float32,cuda:0)', 'reference_extrinsics': 'Tensor([1, 25, 4, 
# 4],torch.float32,cuda:0)', 'reference_intrinsics': 'Tensor([1, 25, 3, 
# 3],torch.float32,cuda:0)'}
# describe(target_latents)='Tensor([1, 2400, 128],torch.bfloat16,cuda:0)' 
# describe(condition_latents)='Tensor([1, 2400, 128],torch.bfloat16,cuda:0)'
# describe(target.get('fps', None))=None
# describe(sigmas)='Tensor([1],torch.float32,cuda:0)'

CustomTrainingBatch = Any
class CustomReferenceVideoTrainingStrategy(TrainingStrategy):
    def pi3_encoder_feature(self, images: Float[torch.Tensor, 'b n c h w']) -> Float[torch.Tensor, 'b n ph pw d']:
        b, n, c, h, w = images.shape # assert c == 3
        images = (images - self._pi3.image_mean) / self._pi3.image_std # type: ignore

        patch_h, patch_w = h // 14, w // 14
        hidden = self._pi3.encoder(rearrange(images, 'b n c h w -> (b n) c h w'), is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden['x_norm_patchtokens']
        return hidden.view(b, n, patch_h, patch_w, -1)

    def pi3_decoder_feature(self, encoder_feat: Float[torch.Tensor, 'b n ph pw d']) -> Float[torch.Tensor, 'b n ph pw e']:
        b, n, ph, pw, d = encoder_feat.shape
        h, w = ph * 14, pw * 14
        hidden, pos = self._pi3.decode(rearrange(encoder_feat, 'b n ph pw d -> (b n) (ph pw) d'), n, h, w)
        return hidden[:, self._pi3.patch_start_idx:].view(b, n, ph, pw, -1) # remove register tokens

    @torch.compile
    def pi3_feature(self, images: Float[torch.Tensor, 'b n c h w']) -> Float[torch.Tensor, 'b n ph pw 2*d']:
        encoder_feat = self.pi3_encoder_feature(images)
        decoder_feat = self.pi3_decoder_feature(encoder_feat)
        return decoder_feat

    def __init__(self, conditioning_config: ConditioningConfig, vae: AutoencoderKLLTXVideo, pi3: Any):
        super().__init__(conditioning_config)
        self._vae = vae

        from pi3.models.pi3 import Pi3
        self._pi3 = cast(Pi3, pi3)
        self._pi3_transform = Pi3Config.transform()

    def get_data_sources(self):
        return {}

    @torch.inference_mode()
    def prepare_batch(self, batch: CustomDL3DV10KDatasetBatch, timestep_sampler: TimestepSampler) -> CustomTrainingBatch:
        t_compress, s_compress = self._vae.temporal_compression_ratio, self._vae.spatial_compression_ratio

        prompt_embeds = batch["prompt_embeds"] # b s=256 c=4096
        prompt_attention_mask = batch["prompt_attention_mask"] # b s=256

        target_pixels: Float[torch.Tensor, 'b t c h w'] = batch['target_pixels'] # ground truth, 30fps, 25 frames
        reference_pixels: Float[torch.Tensor, 'b t c h w'] = batch['reference_pixels'] # reference video for camera control, 30fps; but we will drop 7/8 frames and train the model to interpolate between frames
        condition_pixels: Float[torch.Tensor, 'b f c h w'] = batch['condition_pixels'] # condition video or individual images.

        b, t, c, h, w = target_pixels.shape
        _, f, _, _, _ = condition_pixels.shape

        reference_pixels = reference_pixels[:, [i for i in range(0, t, t_compress)]]
        _, r, _, _, _ = reference_pixels.shape

        latent_frames = (t - 1) // t_compress + 1
        latent_height = h // s_compress
        latent_width = w // s_compress

        # Step 1: get feature
        # Currently, we only use feature from pi3 decoder outputs
        # pi3_input_images = self._pi3_transform(torch.cat([condition_pixels, reference_pixels], dim=1)) # b t+f c h w
        # pi3_input_images = F.interpolate(rearrange(pi3_input_images, 'b t_plus_r c h w -> (b t_plus_r) c h w'), size=(latent_height*14, latent_width*14), mode='bicubic', align_corners=False, antialias=True)
        # feature = self.pi3_feature(rearrange(pi3_input_images.to(self._vae.dtype), '(b t_plus_r) c h w -> b t_plus_r c h w', b=b)) # b 2*d=2*1024 t+f latent_height latent_width

        # Step 2: get vae latents
        # frames from condition_pixels are treated as individual images.
        # reference_pixels are temporally downsampled: [0, 1, ..., 24] -> [0]+[1..8]+[9..16]+[17..24], take 0, 8, 16, 24
        # ic-lora format: [cond[0]] [cond[1]] ... [cond[-1]] [ref[0]] [ref[8]] [ref[16]] [ref[24]] [noise[0]..noise[24]]
        # -- t ->
        # [f tokens] + [r tokens] + [latent_frames tokens]
        # [       feature       ] + [        zeros       ]
        latents_mean, latents_std, scaling_factor = self._vae.latents_mean, self._vae.latents_std, self._vae.config['scaling_factor']
        latents_mean, latents_std = latents_mean.view(1, -1, 1, 1, 1), latents_std.view(1, -1, 1, 1, 1)

        def encode_as_video(pixels: Float[torch.Tensor, 'b f c h w']) -> Float[torch.Tensor, 'b c latent_frames latent_height latent_width']:
            latents = rearrange(pixels, 'b f c h w -> b c f h w').to(self._vae.dtype)
            latents = cast(AutoencoderKLOutput, self._vae.encode(latents*2-1, return_dict=True)).latent_dist.sample()
            return (latents - latents_mean) / latents_std * scaling_factor

        def encode_as_images(pixels: Float[torch.Tensor, 'b f c h w']) -> Float[torch.Tensor, 'b c f h w']:
            latents = rearrange(pixels, 'b f c h w -> (b f) c 1 h w').to(self._vae.dtype)
            latents = cast(AutoencoderKLOutput, self._vae.encode(latents*2-1, return_dict=True)).latent_dist.sample()
            latents = rearrange(latents, '(b f) c n h w -> b c (f n) h w', b=pixels.size(0))
            return (latents - latents_mean) / latents_std * scaling_factor

        target_latents = patchify(encode_as_video(target_pixels)) # b c latent_frames latent_height latent_width
        condition_latents = patchify(encode_as_images(condition_pixels)) # b c f latent_height latent_width
        reference_latents = patchify(encode_as_images(reference_pixels)) # b c r latent_height latent_width
        target_to_reference_latents = patchify(encode_as_images(target_pixels[:, [i for i in range(0, t, t_compress)]]))

        # for an auxilary training target: predictions on the reference latent tokens should be consistent with corresponding frames in target pixels
        # -- t ->      [=tgt2ref]
        # [f tokens] + [r tokens] + [latent_frames tokens]
        # [       feature       ] + [        zeros       ]
        # [any]        [0,8,16,.]   [0,1,9,16,...        ]

        # assemble masks, coords, latents
        with target_latents.device:
            # assemble latents
            conditioning_latents = torch.cat([condition_latents, reference_latents, target_latents], dim=1) # b f+r+t h w c, clean

            sigmas: Float[torch.Tensor, 'b'] = timestep_sampler.sample_for(tokenize(conditioning_latents))
            sampled_timestep_values: Integer[torch.Tensor, 'b'] = torch.round(sigmas * 1000.0).long()

            # assemble coords
            target_indices = patch_indices(target_latents)
            target_coords = indices_to_pixel_coords(target_indices) # Float[torch.Tensor, 'b f h w c=3'], 以像素为单位的 latent 格子坐标

            reference_indices = patch_indices(reference_latents)
            reference_coords = reference_indices * torch.tensor([t_compress, s_compress, s_compress])

            condition_indices = torch.stack(torch.meshgrid(
                torch.arange(-f, 0), torch.arange(latent_height), torch.arange(latent_width), indexing='ij'
            ), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1, 1) # b f h w 3
            condition_coords = condition_indices * torch.tensor([t_compress, s_compress, s_compress])

            fps = torch.tensor(20.0) # override input video fps
            manual_token_coords = torch.cat([condition_coords, reference_coords, target_coords], dim=1) # b f+r+t h w 3
            manual_token_coords = rearrange(manual_token_coords, 'b frt h w c -> b c (frt h w)').float()
            manual_token_coords[:, 0] /= (fps + torch.rand_like(fps) * 10)

            # assemble masks
            target_mask = torch.zeros((b, 1, latent_frames, latent_height, latent_width), dtype=torch.bool)
            target_mask[:, :, 0] = (torch.rand((b, 1, 1, 1)) < self.conditioning_config.first_frame_conditioning_p) # apply random first frame conditioning
            condition_reference_mask = torch.ones((b, 1, f + r, latent_height, latent_width))
            conditioning_mask = torch.cat([condition_reference_mask.float(), target_mask.float()], dim=2) # b 1 f+r+t h w

            # from gshub.utils import describe
            # now, add noise to latents.
            # conditioning_latents is somewhat like the target.
            # with a small probability, all the conditioning latents will be mixed with noise;
            # otherwise, conditioning_mask=1 means not mix, conditioning_mask=0 means mix with weight sigma
            noise = torch.randn_like(conditioning_latents)
            applied_sigmas = torch.min(1 - conditioning_mask, rearrange(sigmas, 'b -> b 1 1 1 1')) # b 1 f+r+t h w
            applied_sigmas = torch.where((torch.rand((b, 1, 1, 1, 1)) < 0.1), sigmas.view(b, 1, 1, 1, 1), applied_sigmas)
            applied_sigmas = rearrange(applied_sigmas.to(conditioning_latents), 'b c t h w -> b t h w c') # c=1, somewhat like patchified conditioning_mask

            # print(f"{applied_sigmas=}")
            noisy_target = torch.lerp(conditioning_latents, noise, applied_sigmas) # as input:hidden_states
            targets = noise - conditioning_latents # as prediction target
            # targets = conditioning_latents - noise # as prediction target

        # {
        #     'hidden_states': 'Tensor([1, 9600, 128],torch.bfloat16,cuda:0)',
        #     'encoder_hidden_states': 'Tensor([1, 256, 4096],torch.float16,cuda:0)',
        #     'encoder_attention_mask': 'Tensor([1, 256],torch.bool,cuda:0)',
        #     'video_coords': 'Tensor([1, 3, 9600],torch.float32,cuda:0)',
        #     'timestep': 'Tensor([1, 9600],torch.bfloat16,cuda:0)',
        #     'targets': 'Tensor([1, 9600, 128],torch.bfloat16,cuda:0)',
        #     'conditioning_latents': 'Tensor([1, 128, 16, 20, 30],torch.bfloat16,cuda:0)',
        #     'conditioning_mask': 'Tensor([1, 1, 16, 20, 30],torch.float32,cuda:0)',
        #     'feature': 'Tensor([1, 12, 20, 30, 2048],torch.bfloat16,cuda:0)',
        # }

        target_seq_len = (latent_frames + r) * latent_height * latent_width
        return {
            'encoder_hidden_states': prompt_embeds,
            'encoder_attention_mask': prompt_attention_mask,

            'video_coords': manual_token_coords[:, :, -target_seq_len:],
            'hidden_states': tokenize(noisy_target)[:, -target_seq_len:],
            'timestep': 1000 * tokenize(applied_sigmas).squeeze(-1)[:, -target_seq_len:],

            # tokenized targets
            'targets': tokenize(targets)[:, -target_seq_len:],
            # unpatchified
            'conditioning_latents': rearrange(conditioning_latents, 'b fp hp wp (c pf ph pw) -> b c (fp pf) (hp ph) (wp pw)', pf=1, ph=1, pw=1), # b c f+r+t h w
            # never patchified
            'conditioning_mask': conditioning_mask[:, :, -latent_frames:], # b 1 f+r+t h w
            # 'feature': feature, # b f+r h w 2048

            'latent_frames': latent_frames,
            'latent_height': latent_height,
            'latent_width': latent_width,
        }

    def prepare_model_inputs(self, batch: CustomTrainingBatch):
        return {
            k: v for k, v in batch.items()
            if k in 'hidden_states encoder_hidden_states encoder_attention_mask video_coords timestep'.split()
        }

    def compute_loss(self, model_pred: Float[torch.Tensor, 'b n c'], batch: CustomTrainingBatch) -> torch.Tensor:
        """Compute masked loss only on target portion, excluding conditioning tokens."""
        # Extract target portion from model prediction and conditioning mask
        latent_frames = batch['latent_frames']
        latent_height = batch['latent_height']
        latent_width = batch['latent_width']
        target_seq_len = latent_frames * latent_height * latent_width

        target_pred = model_pred[:, -target_seq_len:]
        target_conditioning_mask = tokenize(patchify(batch['conditioning_mask']))[:, -target_seq_len:]

        # print(f"{describe(target_pred)=}")
        # print(f"{describe(batch['targets'])=}")

        loss = (target_pred - batch['targets'][:, -target_seq_len:]).pow(2)
        # Create loss mask: exclude conditioning tokens
        loss_mask = (1 - target_conditioning_mask).float()
        # Apply original loss computation pattern
        loss = loss.mul(loss_mask).div(loss_mask.mean())
        return loss.mean()

class CustomVideoTrainingStrategyControlGroup(TrainingStrategy):
    def __init__(self, conditioning_config: ConditioningConfig, vae: AutoencoderKLLTXVideo, pi3: Any):
        super().__init__(conditioning_config)
        self._vae = vae

        from pi3.models.pi3 import Pi3
        self._pi3 = cast(Pi3, pi3)
        self._pi3_transform = Pi3Config.transform()

        from ltxv_trainer.training_strategies import StandardTrainingStrategy
        self.strategy = StandardTrainingStrategy(conditioning_config)

    def get_data_sources(self):
        return {}

    def prepare_batch(self, batch: CustomDL3DV10KDatasetBatch, timestep_sampler: TimestepSampler) -> CustomTrainingBatch:
        t_compress, s_compress = self._vae.temporal_compression_ratio, self._vae.spatial_compression_ratio

        prompt_embeds = batch["prompt_embeds"] # b s=256 c=4096
        prompt_attention_mask = batch["prompt_attention_mask"] # b s=256

        target_pixels = batch['target_pixels']
        b, t, c, h, w = target_pixels.shape
        latent_frames = (t - 1) // t_compress + 1
        latent_height = h // s_compress
        latent_width = w // s_compress

        latents_mean, latents_std, scaling_factor = self._vae.latents_mean, self._vae.latents_std, self._vae.config['scaling_factor']
        latents_mean, latents_std = latents_mean.view(1, -1, 1, 1, 1), latents_std.view(1, -1, 1, 1, 1)
        def encode_as_video(pixels: Float[torch.Tensor, 'b f c h w']) -> Float[torch.Tensor, 'b c latent_frames latent_height latent_width']:
            latents = rearrange(pixels, 'b f c h w -> b c f h w').to(self._vae.dtype)
            latents = cast(AutoencoderKLOutput, self._vae.encode(latents*2-1, return_dict=True)).latent_dist.sample()
            return (latents - latents_mean) / latents_std * scaling_factor

        target_latents = patchify(encode_as_video(target_pixels)) # b c latent_frames latent_height latent_width

        return self.strategy.prepare_batch({
            'latents': {
                # {"latents": latents, "num_frames": num_frames, "height": height, "width": width}
                'latents': tokenize(target_latents),
                'num_frames': torch.tensor(latent_frames)[None].expand(target_latents.size(0)),
                'height': torch.tensor(latent_height)[None].expand(target_latents.size(0)),
                'width': torch.tensor(latent_width)[None].expand(target_latents.size(0)),
                'fps': torch.tensor(10)[None].expand(target_latents.size(0)),
            },
            'conditions': {
                'prompt_embeds': prompt_embeds,
                'prompt_attention_mask': prompt_attention_mask,
            }
        }, timestep_sampler)
    
    def prepare_model_inputs(self, batch: CustomTrainingBatch):
        return self.strategy.prepare_model_inputs(batch)
    
    def compute_loss(self, model_pred: Float[torch.Tensor, 'b n c'], batch: CustomTrainingBatch) -> torch.Tensor:
        return self.strategy.compute_loss(model_pred, batch)

class CustomTrainer(LtxvTrainer):
    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""
        prepare = self._accelerator.prepare
        self._vae = prepare(self._vae) # Note: No .to('cpu')
        self._transformer = prepare(self._transformer)
        self._text_encoder = prepare(self._text_encoder)

        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder = self._text_encoder.to("cpu")

        # Enable gradient checkpointing if requested
        if self._config.optimization.enable_gradient_checkpointing:
            if isinstance(self._transformer, torch.nn.parallel.DistributedDataParallel):
                # If using DDP, enable gradient checkpointing on the wrapped model
                self._transformer.module.enable_gradient_checkpointing()
            else:
                self._transformer.enable_gradient_checkpointing()

    def _compile_transformer(self) -> None:
        """Compile the transformer model with Torch Inductor."""
        super()._compile_transformer()

        # compile_module = functools.partial(torch.compile, mode=self._config.acceleration.compilation_mode)
        # self._pi3.encoder.blocks = nn.ModuleList([compile_module(block) for block in self._pi3.encoder.blocks]) # type: ignore
        # self._pi3.decoder = nn.ModuleList([compile_module(block) for block in self._pi3.decoder]) # type: ignore

    def __init__(self, trainer_config: CustomTrainerConfig) -> None:
        self._config = trainer_config
        self._print_config(trainer_config)
        self._setup_accelerator()
        self._load_models()
        self._compile_transformer()
        self._collect_trainable_params()
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._global_step = -1
        self._checkpoint_paths = []
        self._init_wandb()
        # self._training_strategy = CustomReferenceVideoTrainingStrategy(self._config.conditioning, self._vae, self._pi3)
        self._training_strategy = CustomVideoTrainingStrategyControlGroup(self._config.conditioning, self._vae, self._pi3)

    def _load_models(self) -> None:
        """Load the LTXV model components."""
        # Load all model components using the new loader
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        # Prepare components with accelerator
        self._scheduler = load_scheduler()
        self._tokenizer = load_tokenizer()
        self._text_encoder = load_text_encoder()
        self._vae = load_vae(self._config.model.model_source, dtype=torch.bfloat16)
        self._transformer = load_ltxv_xfmr_2b_manually(LTXVideoTransformer3DModel, torch_dtype=transformer_dtype)
        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")
            logger.warning(f"Quantizing model with precision: {self._config.acceleration.quantization}")
            self._transformer = quantize_model(self._transformer, precision=self._config.acceleration.quantization)

        self._pi3 = self._config.pi3.model().to(torch.bfloat16).eval()
        self._pi3.requires_grad_(False)
        self._pi3.to(self._accelerator.device)

        # Freeze all models. We later unfreeze the transformer based on training mode.
        self._text_encoder.requires_grad_(False)
        self._vae.requires_grad_(False)
        self._transformer.requires_grad_(False)

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            self._dataset = CustomDL3DV10KDataset(
                self._config.data.preprocessed_data_root, num_frames=25, num_cond_frames=8, resolution=(640, 960),
                discrete_reference_indices=False, discrete_condition_indices=True,
            )
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from {self._dataset}")

        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._config.data.num_dataloader_workers,
            pin_memory=self._config.data.num_dataloader_workers > 0,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

    def _training_step(self, batch: CustomDL3DV10KDatasetBatch) -> torch.Tensor:
        """Perform a single training step using the configured strategy."""
        # Use strategy to prepare the training batch
        training_batch = self._training_strategy.prepare_batch(batch, self._timestep_sampler)

        # Use strategy to prepare model inputs
        model_inputs = self._training_strategy.prepare_model_inputs(training_batch)

        # Run transformer forward pass
        # model_pred = self._transformer(**model_inputs, return_dict=False)[0]
        model_pred = self._transformer(**model_inputs)[0]

        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(model_pred, training_batch)

        return loss

    @torch.no_grad()
    @torch.compiler.set_stance("force_eager")
    def _sample_videos(self, progress: Progress) -> Optional[list[Path]]:
        """Run validation by generating images from validation prompts."""
        if hasattr(self, 'config_path'): # reload config
            config_path = getattr(self, 'config_path')
            import yaml
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
            self._config.validation = self._config.__class__(**config_data).validation

        self._vae.to(self._accelerator.device)
        # Model is already in the correct device if loaded in 8-bit.
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to(self._accelerator.device)

        pipeline = BaseLTXVPipeline(
            scheduler=copy.deepcopy(self._scheduler),
            vae=self._accelerator.unwrap_model(self._vae), # type: ignore
            text_encoder=self._accelerator.unwrap_model(self._text_encoder), # type: ignore
            tokenizer=self._tokenizer,
            transformer=self._accelerator.unwrap_model(self._transformer), # type: ignore
        )
        pipeline.set_progress_bar_config(disable=True)

        # Create a task in the sampling progress
        task = progress.add_task("sampling", total=len(self._config.validation.samples))

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        i = 0
        video_paths = []
        for j, sample in enumerate(self._config.validation.samples):
            generator = torch.Generator(device=self._accelerator.device).manual_seed(self._config.validation.seed)

            width, height, frames = self._config.validation.video_dims
            prompts = [not_none(sample.prompt or self._config.validation.default_prompt)]
            negative_prompts = [not_none(sample.negative_prompt or self._config.validation.negative_prompt)]

            prompt_embeds = pipeline.encode_prompts(prompts)
            negative_prompts = pipeline.encode_prompts(negative_prompts)

            batch_size = 1
            latent_channels = 128
            latent_frames = (frames - 1) // 8 + 1
            latent_height = height //32
            latent_width = width // 32

            with self._accelerator.device: # type: ignore
                latents = torch.randn((batch_size, latent_channels, latent_frames, latent_height, latent_width), generator=generator)
                latent_coords = rearrange(indices_to_pixel_coords(patch_indices(patchify(latents))), 'b f h w c -> b c (f h w)').float()
            latent_coords[:, 0] = latent_coords[:, 0] / 25

            if sample.reference_video is not None:
                pass # TODO: requires special treatments defined by self._training_strategy
            if sample.condition_video is not None:
                pass # TODO: requires special treatments defined by self._training_strategy

            if sample.first_frame is not None:
                with self._accelerator.device: # type: ignore
                    condition_latents = torch.randn((batch_size, latent_channels, latent_frames, latent_height, latent_width), generator=generator)
                    condition_mask = torch.zeros((batch_size, 1, latent_frames, latent_height, latent_width))

                if isinstance(sample.first_frame, str):
                    # load first_frame image from an image file
                    first_frames = torch.as_tensor(iio.imread(sample.first_frame)).to(self._accelerator.device).float()[None].expand(-1, -1, -1, -1) / 255
                else:
                    video_decoder = VideoDecoder(sample.first_frame.path, dimension_order='NHWC')
                    first_frames = video_decoder[sample.first_frame.index:sample.first_frame.index+1].to(self._accelerator.device).float() / 255
                first_frames = F.interpolate(first_frames.permute(0,3,1,2), size=(height, width), mode='bicubic', align_corners=False, antialias=True).to(self._vae.dtype) # t c h w
                first_frame_latents = encode_as_video(self._vae, first_frames[None])
                condition_latents[:, :, 0:1] = first_frame_latents[:, :, 0:1]
                condition_mask[:, :, 0:1] = 1
                conditioning = (
                    tokenize(patchify(condition_latents)),
                    tokenize(patchify(condition_mask.expand(-1, 128, -1, -1, -1)))
                )
            else:
                conditioning = None
            image_cond_noise_scale = 0.15

            with torch.amp.autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                # result = pipeline(**pipeline_inputs)
                # videos = result.frames
                # latents = pipeline.sample_latents(**pipeline_inputs, device=self._accelerator.device)
                latents = pipeline.latents_sample_loop(
                    latents=tokenize(patchify(latents)),
                    latent_coords=latent_coords,
                    prompt_embeds=prompt_embeds['prompt_embeds'],
                    prompt_attention_mask=prompt_embeds['prompt_attention_mask'],
                    negative_prompt_embeds=negative_prompts['prompt_embeds'],
                    negative_prompt_attention_mask=negative_prompts['prompt_attention_mask'],
                    num_inference_steps=self._config.validation.inference_steps,
                    conditioning=conditioning,
                )
                latents = rearrange(latents, 'b (fp hp wp) (c pf ph pw) -> b c (fp pf) (hp ph) (wp pw)', fp=latent_frames, hp=latent_height, wp=latent_width, pf=1, ph=1, pw=1)

                videos = pipeline.sample_videos(latents, device=self._accelerator.device)
                # first_frame = pipeline.sample_videos(first_frame_latents, device=self._accelerator.device)

            for video in videos:
                video_path = output_dir / f"step_{self._global_step:06d}_{i}.mp4"
                # export_to_video(video, str(video_path), fps=24)
                torchvision.io.write_video(str(video_path), video.permute(1,2,3,0).clamp(0,1).cpu()*255, fps=24, options={'crf': '20'})

                # video_path = output_dir / f"step_{self._global_step:06d}_{i}_first_frame.mp4"
                # torchvision.io.write_video(str(video_path), first_frame.permute(1,2,3,0).clamp(0,1).cpu()*255, fps=24, options={'crf': '20'})
                video_paths.append(video_path)
                i += 1
            progress.update(task, advance=1)

        progress.remove_task(task)

        # Move unused components back to CPU.
        # self._vae.to("cpu")
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to("cpu")

        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        logger.info(f"🎥 Validation samples for step {self._global_step} saved in {rel_outputs_path}")
        return video_paths

    @torch.no_grad()
    @torch.compiler.set_stance("force_eager")
    def _sample_videos_old(self, progress: Progress) -> Optional[list[Path]]:
        """Run validation by generating images from validation prompts."""
        if hasattr(self, 'config_path'):
            config_path = getattr(self, 'config_path')
            import yaml
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
            self._config.validation = self._config.__class__(**config_data).validation

        self._vae.to(self._accelerator.device)
        # Model is already in the correct device if loaded in 8-bit.
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to(self._accelerator.device)

        pipeline = BaseLTXVPipeline(
            scheduler=copy.deepcopy(self._scheduler),
            vae=self._accelerator.unwrap_model(self._vae), # type: ignore
            text_encoder=self._accelerator.unwrap_model(self._text_encoder), # type: ignore
            tokenizer=self._tokenizer,
            transformer=self._accelerator.unwrap_model(self._transformer), # type: ignore
        )
        pipeline.set_progress_bar_config(disable=True)

        # Create a task in the sampling progress
        task = progress.add_task("sampling", total=len(self._config.validation.samples))

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        i = 0
        video_paths = []
        for j, sample in enumerate(self._config.validation.samples):
            generator = torch.Generator(device=self._accelerator.device).manual_seed(self._config.validation.seed)

            width, height, frames = self._config.validation.video_dims
            prompt = sample.prompt or self._config.validation.default_prompt
            negative_prompt = sample.negative_prompt or self._config.validation.negative_prompt

            pipeline_inputs = {
                "prompts": [prompt],
                "negative_prompts": [negative_prompt],
                "width": width,
                "height": height,
                "num_frames": frames,
                "num_inference_steps": self._config.validation.inference_steps,
                "cfg_source": self._config.validation.cfg_scale,
                "stg_source": self._config.validation.stg_scale,
                "rescaling_source": self._config.validation.rescaling_scale,
                "generator": generator,
                # "output_reference_comparison": True,
                # "dinov3": self._dinov3,
            }

            batch_size = 1
            latent_channels = 128
            latent_frames = (frames - 1) // 8 + 1
            latent_height = height //32
            latent_width = width // 32

            if sample.reference_video is not None:
                pass # TODO: requires special treatments defined by self._training_strategy
            if sample.condition_video is not None:
                pass # TODO: requires special treatments defined by self._training_strategy

            if sample.first_frame is not None:
                with self._accelerator.device: # type: ignore
                    condition_latents = torch.randn((batch_size, latent_channels, latent_frames, latent_height, latent_width), generator=generator)
                    condition_mask = torch.zeros((batch_size, 1, latent_frames, latent_height, latent_width))

                if isinstance(sample.first_frame, str):
                    # load first_frame image from an image file
                    first_frames = torch.as_tensor(iio.imread(sample.first_frame)).to(self._accelerator.device).float()[None].expand(-1, -1, -1, -1) / 255
                else:
                    video_decoder = VideoDecoder(sample.first_frame.path, dimension_order='NHWC')
                    first_frames = video_decoder[:].to(self._accelerator.device).float() / 255
                first_frames = F.interpolate(first_frames.permute(0,3,1,2), size=(height, width), mode='bicubic', align_corners=False, antialias=True).to(self._vae.dtype)[0:1] # t c h w
                with torch.amp.autocast(self._accelerator.device.type, dtype=self._vae.dtype):
                    first_frame_latents = cast(AutoencoderKLOutput, self._vae.encode(rearrange(first_frames, 'f c h w -> 1 c f h w')*2-1)).latent_dist.sample()

                latents_mean, latents_std, scaling_factor = self._vae.latents_mean.to(first_frame_latents), self._vae.latents_std.to(first_frame_latents), self._vae.config['scaling_factor']
                latents_mean, latents_std = latents_mean.view(1, -1, 1, 1, 1), latents_std.view(1, -1, 1, 1, 1)
                normalized_first_frame_latents = (first_frame_latents - latents_mean) / latents_std * scaling_factor

                condition_latents[:, :, 0:1] = normalized_first_frame_latents[:, :, 0:1]
                condition_mask[:, :, 0:1] = 1
                conditioning = (condition_latents, condition_mask)
            else:
                conditioning = None
            pipeline_inputs['conditioning'] = conditioning
            pipeline_inputs['image_cond_noise_scale'] = 0.15

            # # Load and add reference video, if provided
            # if self._config.validation.reference_videos is not None:
            #     assert self._config.validation.condition_videos is not None
            #     ref_video, _ = read_video(self._config.validation.reference_videos[j])[-frames:]
            #     cond_video, _ = read_video(self._config.validation.condition_videos[j])[-frames:]
            #     pipeline_inputs["reference_video"] = (cond_video, ref_video)
            # # Load and add first frame image, if provided
            # if use_images:
            #     assert self._config.validation.images is not None
            #     image_path = self._config.validation.images[j]
            #     if image_path is None:
            #         image = ref_video[0]
            #         current_height, current_width = image.shape[1:]
            #         aspect_ratio = current_width / current_height
            #         target_aspect_ratio = width / height
            #         if aspect_ratio > target_aspect_ratio:
            #             # Width is relatively larger, resize based on height
            #             resize_height = height
            #             resize_width = int(resize_height * aspect_ratio)
            #         else:
            #             # Height is relatively larger, resize based on width
            #             resize_width = width
            #             resize_height = int(resize_width / aspect_ratio)
            #         image = torchvision.transforms.functional.resize(image, [resize_height, resize_width], antialias=True)
            #         image = torchvision.transforms.functional.center_crop(image, [height, width])
            #         pipeline_inputs["image"] = image
            #     else:
            #         if Path(image_path).is_file():
            #             image = open_image_as_srgb(image_path)
            #             if image.size != (height, width):
            #                 # Resize and center crop the image to match the validation video dimensions
            #                 image = F.resize(image, size=min(width, height))
            #                 image = F.center_crop(image, output_size=(width, height))
            #             pipeline_inputs["image"] = image

            with torch.amp.autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                # result = pipeline(**pipeline_inputs)
                # videos = result.frames
                latents = pipeline.sample_latents(**pipeline_inputs, device=self._accelerator.device)
                videos = pipeline.sample_videos(latents, device=self._accelerator.device)
                # first_frame = pipeline.sample_videos(first_frame_latents, device=self._accelerator.device)

            for video in videos:
                video_path = output_dir / f"step_{self._global_step:06d}_{i}.mp4"
                # export_to_video(video, str(video_path), fps=24)
                torchvision.io.write_video(str(video_path), video.permute(1,2,3,0).clamp(0,1).cpu()*255, fps=24, options={'crf': '20'})

                # video_path = output_dir / f"step_{self._global_step:06d}_{i}_first_frame.mp4"
                # torchvision.io.write_video(str(video_path), first_frame.permute(1,2,3,0).clamp(0,1).cpu()*255, fps=24, options={'crf': '20'})
                video_paths.append(video_path)
                i += 1
            progress.update(task, advance=1)

        progress.remove_task(task)

        # Move unused components back to CPU.
        # self._vae.to("cpu")
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to("cpu")

        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        logger.info(f"🎥 Validation samples for step {self._global_step} saved in {rel_outputs_path}")
        return video_paths


##### Pi3 probe

class CustomPi3EncoderProbeVideoTrainingStrategy(TrainingStrategy):
    @torch.compile
    def pi3_encoder_feature(self, images: Float[torch.Tensor, 'b n c h w']) -> Float[torch.Tensor, 'b n ph pw d']:
        b, n, c, h, w = images.shape # assert c == 3
        images = (images - self._pi3.image_mean) / self._pi3.image_std # type: ignore

        patch_h, patch_w = h // 14, w // 14
        hidden = self._pi3.encoder(rearrange(images, 'b n c h w -> (b n) c h w'), is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden['x_norm_patchtokens']
        return hidden.view(b, n, patch_h, patch_w, -1)

    def pi3_decoder_feature(self, encoder_feat: Float[torch.Tensor, 'b n ph pw d']) -> Float[torch.Tensor, 'b n ph pw e']:
        b, n, ph, pw, d = encoder_feat.shape
        h, w = ph * 14, pw * 14
        hidden, pos = self._pi3.decode(rearrange(encoder_feat, 'b n ph pw d -> (b n) (ph pw) d'), n, h, w)
        return hidden[:, self._pi3.patch_start_idx:].view(b, n, ph, pw, -1) # remove register tokens

    def pi3_feature(self, images: Float[torch.Tensor, 'b n c h w']) -> Float[torch.Tensor, 'b n ph pw 2*d']:
        encoder_feat = self.pi3_encoder_feature(images)
        decoder_feat = self.pi3_decoder_feature(encoder_feat)
        return decoder_feat

    def __init__(self, conditioning_config: ConditioningConfig, vae: AutoencoderKLLTXVideo, pi3: Any):
        super().__init__(conditioning_config)
        self._vae = vae

        from pi3.models.pi3 import Pi3
        self._pi3 = cast(Pi3, pi3)
        self._pi3_transform = Pi3Config.transform()

    def get_data_sources(self):
        return {}

    @torch.inference_mode()
    def prepare_batch(self, batch: CustomDL3DV10KDatasetBatch, timestep_sampler: TimestepSampler) -> CustomTrainingBatch:
        t_compress, s_compress = self._vae.temporal_compression_ratio, self._vae.spatial_compression_ratio

        prompt_embeds = batch["prompt_embeds"] # b s=256 c=4096
        prompt_attention_mask = batch["prompt_attention_mask"] # b s=256

        target_pixels: Float[torch.Tensor, 'b t c h w'] = batch['target_pixels'] # ground truth, 30fps, 25 frames
        # reference_pixels: Float[torch.Tensor, 'b t c h w'] = batch['reference_pixels'] # reference video for camera control, 30fps; but we will drop 7/8 frames and train the model to interpolate between frames
        # condition_pixels: Float[torch.Tensor, 'b f c h w'] = batch['condition_pixels'] # condition video or individual images.

        b, t, c, h, w = target_pixels.shape
        # _, f, _, _, _ = condition_pixels.shape

        # reference_pixels = reference_pixels[:, [i for i in range(0, t, t_compress)]]
        # _, r, _, _, _ = reference_pixels.shape

        latent_frames = (t - 1) // t_compress + 1
        latent_height = h // s_compress
        latent_width = w // s_compress

        # Step 1: get feature
        # Currently, we only use feature from pi3 decoder outputs
        pi3_input_images = self._pi3_transform(target_pixels[:, [max(i-t_compress+1, 0) for i in range(0, t+t_compress-1, t_compress)]]) # b t c h w
        pi3_input_images = F.interpolate(rearrange(pi3_input_images, 'b t_plus_r c h w -> (b t_plus_r) c h w'), size=(latent_height*14, latent_width*14), mode='bicubic', align_corners=False, antialias=True)
        # feature = self.pi3_feature(rearrange(pi3_input_images.to(self._vae.dtype), '(b t_plus_r) c h w -> b t_plus_r c h w', b=b)) # b 2*d=2*1024 t+f latent_height latent_width
        # feature = self.pi3_encoder_feature(rearrange(pi3_input_images.to(self._vae.dtype), '(b t_plus_r) c h w -> b t_plus_r c h w', b=b)) # b t latent_height latent_width c
        feature = self.pi3_feature(rearrange(pi3_input_images.to(self._vae.dtype), '(b t_plus_r) c h w -> b t_plus_r c h w', b=b)) # b t latent_height latent_width c

        # Step 2: get vae latents
        # frames from condition_pixels are treated as individual images.
        # reference_pixels are temporally downsampled: [0, 1, ..., 24] -> [0]+[1..8]+[9..16]+[17..24], take 0, 8, 16, 24
        # ic-lora format: [cond[0]] [cond[1]] ... [cond[-1]] [ref[0]] [ref[8]] [ref[16]] [ref[24]] [noise[0]..noise[24]]
        # -- t ->
        # [f tokens] + [r tokens] + [latent_frames tokens]
        # [       feature       ] + [        zeros       ]

        target_latents = patchify(encode_as_video(self._vae, target_pixels)) # b c latent_frames latent_height latent_width
        # condition_latents = patchify(encode_as_images(condition_pixels)) # b c f latent_height latent_width
        # reference_latents = patchify(encode_as_images(reference_pixels)) # b c r latent_height latent_width
        # target_to_reference_latents = patchify(encode_as_images(target_pixels[:, [i for i in range(0, t, t_compress)]]))

        # for an auxilary training target: predictions on the reference latent tokens should be consistent with corresponding frames in target pixels
        # -- t ->      [=tgt2ref]
        # [f tokens] + [r tokens] + [latent_frames tokens]
        # [       feature       ] + [        zeros       ]
        # [any]        [0,8,16,.]   [0,1,9,16,...        ]

        # assemble masks, coords, latents
        with target_latents.device:
            # assemble latents
            conditioning_latents = torch.cat([target_latents], dim=1) # b f+r+t h w c, clean

            sigmas: Float[torch.Tensor, 'b'] = timestep_sampler.sample_for(tokenize(conditioning_latents))
            sampled_timestep_values: Integer[torch.Tensor, 'b'] = torch.round(sigmas * 1000.0).long()

            # assemble coords
            target_indices = patch_indices(target_latents)
            target_coords = indices_to_pixel_coords(target_indices) # Float[torch.Tensor, 'b f h w c=3'], 以像素为单位的 latent 格子坐标

            # reference_indices = patch_indices(reference_latents)
            # reference_coords = reference_indices * torch.tensor([t_compress, s_compress, s_compress])

            # condition_indices = torch.stack(torch.meshgrid(
            #     torch.arange(-f, 0), torch.arange(latent_height), torch.arange(latent_width), indexing='ij'
            # ), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1, 1) # b f h w 3
            # condition_coords = condition_indices * torch.tensor([t_compress, s_compress, s_compress])

            fps = torch.tensor(10.0) # override input video fps
            manual_token_coords = torch.cat([target_coords], dim=1) # b f+r+t h w 3
            manual_token_coords = rearrange(manual_token_coords, 'b frt h w c -> b c (frt h w)').float()
            manual_token_coords[:, 0] /= (fps + torch.rand_like(fps) * 10)

            # assemble masks
            target_mask = torch.zeros((b, 1, latent_frames, latent_height, latent_width), dtype=torch.bool)
            target_mask[:, :, 0] = (torch.rand((b, 1, 1, 1)) < self.conditioning_config.first_frame_conditioning_p) # apply random first frame conditioning
            # condition_reference_mask = torch.ones((b, 1, f + r, latent_height, latent_width))
            conditioning_mask = torch.cat([target_mask.float()], dim=2) # b 1 f+r+t h w

            # from gshub.utils import describe
            # now, add noise to latents.
            # conditioning_latents is somewhat like the target.
            # with a small probability, all the conditioning latents will be mixed with noise;
            # otherwise, conditioning_mask=1 means not mix, conditioning_mask=0 means mix with weight sigma
            noise = torch.randn_like(conditioning_latents)
            applied_sigmas = torch.min(1 - conditioning_mask, rearrange(sigmas, 'b -> b 1 1 1 1')) # b 1 f+r+t h w
            applied_sigmas = torch.where((torch.rand((b, 1, 1, 1, 1)) < 0.1), sigmas.view(b, 1, 1, 1, 1), applied_sigmas)
            applied_sigmas = rearrange(applied_sigmas.to(conditioning_latents), 'b c t h w -> b t h w c') # c=1, somewhat like patchified conditioning_mask

            # print(f"{applied_sigmas=}")
            noisy_target = torch.lerp(conditioning_latents, noise, applied_sigmas) # as input:hidden_states
            targets = noise - conditioning_latents # as prediction target
            # targets = conditioning_latents - noise # as prediction target

        # {
        #     'hidden_states': 'Tensor([1, 9600, 128],torch.bfloat16,cuda:0)',
        #     'encoder_hidden_states': 'Tensor([1, 256, 4096],torch.float16,cuda:0)',
        #     'encoder_attention_mask': 'Tensor([1, 256],torch.bool,cuda:0)',
        #     'video_coords': 'Tensor([1, 3, 9600],torch.float32,cuda:0)',
        #     'timestep': 'Tensor([1, 9600],torch.bfloat16,cuda:0)',
        #     'targets': 'Tensor([1, 9600, 128],torch.bfloat16,cuda:0)',
        #     'conditioning_latents': 'Tensor([1, 128, 16, 20, 30],torch.bfloat16,cuda:0)',
        #     'conditioning_mask': 'Tensor([1, 1, 16, 20, 30],torch.float32,cuda:0)',
        #     'feature': 'Tensor([1, 12, 20, 30, 2048],torch.bfloat16,cuda:0)',
        # }

        target_seq_len = (latent_frames + 0) * latent_height * latent_width
        return {
            'encoder_hidden_states': prompt_embeds,
            'encoder_attention_mask': prompt_attention_mask,

            'video_coords': manual_token_coords[:, :, -target_seq_len:],
            'hidden_states': tokenize(noisy_target)[:, -target_seq_len:],
            'timestep': 1000 * tokenize(applied_sigmas).squeeze(-1)[:, -target_seq_len:],

            # tokenized targets
            'targets': tokenize(targets)[:, -target_seq_len:],
            # unpatchified
            'conditioning_latents': rearrange(conditioning_latents, 'b fp hp wp (c pf ph pw) -> b c (fp pf) (hp ph) (wp pw)', pf=1, ph=1, pw=1), # b c f+r+t h w
            # never patchified
            'conditioning_mask': conditioning_mask[:, :, -latent_frames:], # b 1 f+r+t h w
            'feature': feature, # b t h w 1024

            'latent_frames': latent_frames,
            'latent_height': latent_height,
            'latent_width': latent_width,
        }

    def prepare_model_inputs(self, batch: CustomTrainingBatch):
        model_inputs = {
            k: v for k, v in batch.items()
            if k in 'hidden_states encoder_hidden_states encoder_attention_mask video_coords timestep'.split()
        }
        model_inputs['latents_add'] = rearrange(batch['feature'], 'b t h w d -> b (t h w) d')
        return model_inputs

    def compute_loss(self, model_pred: Float[torch.Tensor, 'b n c'], batch: CustomTrainingBatch) -> torch.Tensor:
        """Compute masked loss only on target portion, excluding conditioning tokens."""
        # Extract target portion from model prediction and conditioning mask
        latent_frames = batch['latent_frames']
        latent_height = batch['latent_height']
        latent_width = batch['latent_width']
        target_seq_len = latent_frames * latent_height * latent_width

        target_pred = model_pred[:, -target_seq_len:]
        target_conditioning_mask = tokenize(patchify(batch['conditioning_mask']))[:, -target_seq_len:]

        # print(f"{describe(target_pred)=}")
        # print(f"{describe(batch['targets'])=}")

        loss = (target_pred - batch['targets'][:, -target_seq_len:]).pow(2)
        # Create loss mask: exclude conditioning tokens
        loss_mask = (1 - target_conditioning_mask).float()
        # Apply original loss computation pattern
        loss = loss.mul(loss_mask).div(loss_mask.mean())
        return loss.mean()

class CustomPi3EncoderProbeTrainer(LtxvTrainer):
    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""
        prepare = self._accelerator.prepare
        self._vae = prepare(self._vae) # Note: No .to('cpu')
        self._transformer = prepare(self._transformer)
        self._text_encoder = prepare(self._text_encoder)

        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder = self._text_encoder.to("cpu")

        # Enable gradient checkpointing if requested
        if self._config.optimization.enable_gradient_checkpointing:
            if isinstance(self._transformer, torch.nn.parallel.DistributedDataParallel):
                # If using DDP, enable gradient checkpointing on the wrapped model
                self._transformer.module.enable_gradient_checkpointing()
            else:
                self._transformer.enable_gradient_checkpointing()

    def _compile_transformer(self) -> None:
        """Compile the transformer model with Torch Inductor."""
        super()._compile_transformer()

        # compile_module = functools.partial(torch.compile, mode=self._config.acceleration.compilation_mode)
        # self._pi3.encoder.blocks = nn.ModuleList([compile_module(block) for block in self._pi3.encoder.blocks]) # type: ignore
        # self._pi3.decoder = nn.ModuleList([compile_module(block) for block in self._pi3.decoder]) # type: ignore

    def __init__(self, trainer_config: CustomTrainerConfig) -> None:
        self._config = trainer_config
        self._print_config(trainer_config)
        self._setup_accelerator()
        self._load_models()
        self._compile_transformer()
        self._collect_trainable_params()
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._global_step = -1
        self._checkpoint_paths = []
        self._init_wandb()
        self._training_strategy = CustomPi3EncoderProbeVideoTrainingStrategy(self._config.conditioning, self._vae, self._pi3)

    def _load_models(self) -> None:
        """Load the LTXV model components."""
        # Load all model components using the new loader
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        # Prepare components with accelerator
        self._scheduler = load_scheduler()
        self._tokenizer = load_tokenizer()
        self._text_encoder = load_text_encoder()
        self._vae = load_vae(self._config.model.model_source, dtype=torch.bfloat16)
        self._transformer = load_ltxv_xfmr_2b_manually(LTXVideoTransformer3DModel, torch_dtype=transformer_dtype, latents_add_in_channels=2048)
        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")
            logger.warning(f"Quantizing model with precision: {self._config.acceleration.quantization}")
            self._transformer = quantize_model(self._transformer, precision=self._config.acceleration.quantization)

        self._pi3 = self._config.pi3.model().to(torch.bfloat16).eval()
        self._pi3.requires_grad_(False)
        self._pi3.to(self._accelerator.device)

        # Freeze all models. We later unfreeze the transformer based on training mode.
        self._text_encoder.requires_grad_(False)
        self._vae.requires_grad_(False)
        self._transformer.requires_grad_(False)

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode."""
        if self._config.model.training_mode == "lora":
            # For LoRA training, first set up LoRA layers
            self._setup_lora()
        elif self._config.model.training_mode == "full":
            # For full training, unfreeze all transformer parameters
            self._transformer.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")
        self._transformer.latents_add_proj_in.requires_grad_(True)

        # self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        self._trainable_params = [p for p in self._transformer.latents_add_proj_in.parameters() if p.requires_grad]
        logger.debug(f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            self._dataset = CustomDL3DV10KDataset(
                self._config.data.preprocessed_data_root, num_frames=25, num_cond_frames=8, resolution=(640, 960),
                discrete_reference_indices=False, discrete_condition_indices=True,
            )
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from {self._dataset}")

        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._config.data.num_dataloader_workers,
            pin_memory=self._config.data.num_dataloader_workers > 0,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

    def _training_step(self, batch: CustomDL3DV10KDatasetBatch) -> torch.Tensor:
        """Perform a single training step using the configured strategy."""
        # Use strategy to prepare the training batch
        training_batch = self._training_strategy.prepare_batch(batch, self._timestep_sampler)

        # Use strategy to prepare model inputs
        model_inputs = self._training_strategy.prepare_model_inputs(training_batch)

        # Run transformer forward pass
        # model_pred = self._transformer(**model_inputs, return_dict=False)[0]
        model_pred = self._transformer(**model_inputs)[0]

        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(model_pred, training_batch)

        return loss

    @torch.no_grad()
    @torch.compiler.set_stance("force_eager")
    def _sample_videos(self, progress: Progress) -> Optional[list[Path]]:
        """Run validation by generating images from validation prompts."""
        if hasattr(self, 'config_path'): # reload config
            config_path = getattr(self, 'config_path')
            import yaml
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
            self._config.validation = self._config.__class__(**config_data).validation

        self._vae.to(self._accelerator.device)
        # Model is already in the correct device if loaded in 8-bit.
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to(self._accelerator.device)

        pipeline = BaseLTXVPipeline(
            scheduler=copy.deepcopy(self._scheduler),
            vae=self._accelerator.unwrap_model(self._vae), # type: ignore
            text_encoder=self._accelerator.unwrap_model(self._text_encoder), # type: ignore
            tokenizer=self._tokenizer,
            transformer=self._accelerator.unwrap_model(self._transformer), # type: ignore
        )
        pipeline.set_progress_bar_config(disable=True)

        # Create a task in the sampling progress
        task = progress.add_task("sampling", total=len(self._config.validation.samples))

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        i = 0
        video_paths = []
        for j, sample in enumerate(self._config.validation.samples):
            generator = torch.Generator(device=self._accelerator.device).manual_seed(self._config.validation.seed)

            width, height, frames = self._config.validation.video_dims
            prompts = [not_none(sample.prompt or self._config.validation.default_prompt)]
            negative_prompts = [not_none(sample.negative_prompt or self._config.validation.negative_prompt)]

            prompt_embeds = pipeline.encode_prompts(prompts)
            negative_prompts = pipeline.encode_prompts(negative_prompts)

            batch_size = 1
            latent_channels = 128
            latent_frames = (frames - 1) // 8 + 1
            latent_height = height //32
            latent_width = width // 32

            with self._accelerator.device: # type: ignore
                latents = torch.randn((batch_size, latent_channels, latent_frames, latent_height, latent_width), generator=generator)
                latent_coords = rearrange(indices_to_pixel_coords(patch_indices(patchify(latents))), 'b f h w c -> b c (f h w)').float()
            latent_coords[:, 0] = latent_coords[:, 0] / 25

            if sample.reference_video is not None:
                pass # TODO: requires special treatments defined by self._training_strategy
            if sample.condition_video is not None:
                pass # TODO: requires special treatments defined by self._training_strategy

            if sample.first_frame is not None:
                with self._accelerator.device: # type: ignore
                    condition_latents = torch.randn((batch_size, latent_channels, latent_frames, latent_height, latent_width), generator=generator)
                    condition_mask = torch.zeros((batch_size, 1, latent_frames, latent_height, latent_width))

                if isinstance(sample.first_frame, str):
                    # load first_frame image from an image file
                    first_frames = torch.as_tensor(iio.imread(sample.first_frame)).to(self._accelerator.device).float()[None].expand(-1, -1, -1, -1) / 255
                else:
                    video_decoder = VideoDecoder(sample.first_frame.path, dimension_order='NHWC')
                    first_frames = video_decoder[sample.first_frame.index:sample.first_frame.index+1].to(self._accelerator.device).float() / 255
                first_frames = F.interpolate(first_frames.permute(0,3,1,2), size=(height, width), mode='bicubic', align_corners=False, antialias=True).to(self._vae.dtype) # t c h w
                first_frame_latents = encode_as_video(self._vae, first_frames[None])
                condition_latents[:, :, 0:1] = first_frame_latents[:, :, 0:1]
                condition_mask[:, :, 0:1] = 1
                conditioning = (
                    tokenize(patchify(condition_latents)),
                    tokenize(patchify(condition_mask.expand(-1, 128, -1, -1, -1)))
                )
            else:
                conditioning = None
            image_cond_noise_scale = 0.15

            with torch.amp.autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                # result = pipeline(**pipeline_inputs)
                # videos = result.frames
                # latents = pipeline.sample_latents(**pipeline_inputs, device=self._accelerator.device)
                latents = pipeline.latents_sample_loop(
                    latents=tokenize(patchify(latents)),
                    latent_coords=latent_coords,
                    prompt_embeds=prompt_embeds['prompt_embeds'],
                    prompt_attention_mask=prompt_embeds['prompt_attention_mask'],
                    negative_prompt_embeds=negative_prompts['prompt_embeds'],
                    negative_prompt_attention_mask=negative_prompts['prompt_attention_mask'],
                    num_inference_steps=self._config.validation.inference_steps,
                    conditioning=conditioning,
                )
                latents = rearrange(latents, 'b (fp hp wp) (c pf ph pw) -> b c (fp pf) (hp ph) (wp pw)', fp=latent_frames, hp=latent_height, wp=latent_width, pf=1, ph=1, pw=1)

                videos = pipeline.sample_videos(latents, device=self._accelerator.device)
                # first_frame = pipeline.sample_videos(first_frame_latents, device=self._accelerator.device)

            for video in videos:
                video_path = output_dir / f"step_{self._global_step:06d}_{i}.mp4"
                # export_to_video(video, str(video_path), fps=24)
                torchvision.io.write_video(str(video_path), video.permute(1,2,3,0).clamp(0,1).cpu()*255, fps=24, options={'crf': '20'})

                # video_path = output_dir / f"step_{self._global_step:06d}_{i}_first_frame.mp4"
                # torchvision.io.write_video(str(video_path), first_frame.permute(1,2,3,0).clamp(0,1).cpu()*255, fps=24, options={'crf': '20'})
                video_paths.append(video_path)
                i += 1
            progress.update(task, advance=1)

        progress.remove_task(task)

        # Move unused components back to CPU.
        # self._vae.to("cpu")
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to("cpu")

        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        logger.info(f"🎥 Validation samples for step {self._global_step} saved in {rel_outputs_path}")
        return video_paths


if __name__ == "__main__":

    import typer
    import yaml

    from rich.console import Console
    console = Console()
    app = typer.Typer(
        pretty_exceptions_enable=False,
        no_args_is_help=True,
        help="Train LTXV models using configuration from YAML files.",
    )

    @app.command()
    def main(config_path: str = typer.Argument(..., help="Path to YAML configuration file")) -> None:
        """Train the model using the provided configuration file."""
        # Load the configuration from the YAML file
        config_path = Path(config_path) # type: ignore
        if not config_path.exists(): # type: ignore
            typer.echo(f"Error: Configuration file {config_path} does not exist.")
            raise typer.Exit(code=1)

        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)

        # Convert the loaded data to the LtxvTrainerConfig object
        try:
            trainer_config = CustomTrainerConfig(**config_data)
        except Exception as e:
            typer.echo(f"Error: Invalid configuration data: {e}")
            raise typer.Exit(code=1) from e

        # Initialize the training process
        # trainer = CustomTrainer(trainer_config)
        trainer = CustomPi3EncoderProbeTrainer(trainer_config)
        setattr(trainer, 'config_path', config_path)
        sample_progress = Progress(
            TextColumn("Sampling validation videos"),
            MofNCompleteColumn(),
            BarColumn(bar_width=40, style="blue"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(compact=True),
        )
        # trainer._sample_videos(sample_progress)

        # from gshub.utils import describe
        # trainer._init_dataloader()
        # trainer._init_timestep_sampler()
        # batch = next(iter(trainer._dataloader))
        # print(f"{describe(batch)=}")
        # print(f"{describe(trainer._training_strategy.prepare_batch(batch, trainer._timestep_sampler))=}")
        # with torch.no_grad():
        #     print(f"{trainer._training_step(batch).item()=}")
        trainer.train()
    app()
