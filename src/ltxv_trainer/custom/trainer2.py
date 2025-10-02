from __future__ import annotations

from abc import abstractmethod
import math
import copy
import random
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional

from tqdm.auto import tqdm, trange
from gshub.utils import describe, not_none, not_none_or_dotenv, read_json, save_json

from einops import *
from typing import *
from typing_extensions import *
from jaxtyping import Float, Shaped, Int, Integer, Bool
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder
from safetensors.torch import load_file, save_file
from peft import LoraConfig, get_peft_model_state_dict
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
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
from ltxv_trainer.model_loader import LtxvModelVersion, load_vae
from ltxv_trainer.custom.base import CustomDL3DV10KDatasetBatch, CustomDL3DV10KDataset, load_ltxv_xfmr_2b_manually, ltxv_resize_and_crop

from ltxv_trainer.custom.inference import BaseLTXVPipeline, LTXVideoTransformer3DModel, Latents, LatentConditioning, TokenizedLatents, TokenizedLatentMask, TokenizedVideoCoords, latent_indices, patch_indices, indices_to_pixel_coords, pixel_to_video_coords, patchify, tokenize, encode_as_video, encode_as_images, randn_like

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

    reference_video: Optional[str | Video] = None
    condition_video: Optional[str | Video] = None
    target_video: Optional[str | Video] = None
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
                ]
                # + [
                #     f'decoder.{i}.attn.qkv' for i in range(36)
                # ] + [
                #     f'decoder.{i}.attn.proj' for i in range(36)
                # ]
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

# region: deprecated
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

    @torch.no_grad()
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
# endregion

CustomBaseTrainingStrategyContext = TypedDict('CustomBaseTrainingStrategyContext', {
    # model inputs:
    'hidden_states': TokenizedLatents,
    'timestep': Shaped[torch.Tensor, 'batch_size num_tokens'],
    'video_coords': TokenizedVideoCoords,

    # train / inference auxiliary inputs:
    'condition_mask': TokenizedLatentMask,
    'model_pred': Optional[TokenizedLatents],
})

class CustomBaseTrainingStrategy(TrainingStrategy):
    def get_data_sources(self):
        # No need to implement; the trainer will handle dataset directly.
        return {}

    @property
    def _override_frame_rate(self) -> Optional[float]:
        return getattr(self, 'override_frame_rate', None)

    @property
    def _default_frame_rate(self) -> float:
        return getattr(self, 'default_frame_rate', 25)

    def compute_loss(self, model_pred: Float[torch.Tensor, 'b n c'], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute masked loss only on target portion, excluding conditioning tokens."""
        loss = (model_pred - batch['model_pred']).pow(2)
        # Create loss mask: exclude conditioning tokens
        loss_mask = (1 - batch['condition_mask']).float()
        # Apply original loss computation pattern
        loss = loss.mul(loss_mask).div(loss_mask.mean() + 1e-3)
        return loss.mean()

    @torch.no_grad()
    def prepare_model_inputs(self, batch: Dict[str, torch.Tensor]):
        model_inputs = {
            k: v for k, v in batch.items()
            if k in 'hidden_states encoder_hidden_states encoder_attention_mask video_coords timestep latents_add'.split()
        }
        return model_inputs

    @torch.no_grad()
    def prepare_batch(self, batch: Dict[str, torch.Tensor], timestep_sampler: TimestepSampler):
        prompt_embeds = batch["prompt_embeds"] # b s=256 c=4096
        prompt_attention_mask = batch["prompt_attention_mask"] # b s=256

        target_pixels: Float[torch.Tensor, 'b t c h w'] = batch['target_pixels']
        reference_pixels: Optional[Float[torch.Tensor, 'b r c h w']] = batch.get('reference_pixels', None)
        condition_pixels: Optional[Float[torch.Tensor, 'b n c h w']] = batch.get('condition_pixels', None)

        b, t, c, h, w = target_pixels.shape

        context = self.prepare_context(
            target_pixels=target_pixels,
            reference_pixels=reference_pixels,
            condition_pixels=condition_pixels,
            video_dims=(w, h, t),
            generator=None,
            timestep_sampler=timestep_sampler,
            frame_rate=self._override_frame_rate or batch.get('fps', self._default_frame_rate)
        )

        return {
            'hidden_states': context['hidden_states'],
            'encoder_hidden_states': prompt_embeds,
            'encoder_attention_mask': prompt_attention_mask,
            'timestep': context['timestep'],
            'video_coords': context['video_coords'],
            'latents_add': context.get('latents_add', None), # b (t h w) (1024 or 2048)

            'model_pred': context['model_pred'], # b (t h w) c
            'condition_mask': context['condition_mask'], # b (t h w) c
            # 'noise': context['noise'], # b (t h w) c
        }

    @abstractmethod
    def prepare_context(
        self,
        *,
        target_pixels: Optional[Float[torch.Tensor, 'b t c h w']] = None,
        reference_pixels: Optional[Float[torch.Tensor, 'b r c h w']] = None,
        condition_pixels: Optional[Float[torch.Tensor, 'b n c h w']] = None,
        first_frame_pixels: Optional[Float[torch.Tensor, 'b f c h w']] = None,
        video_dims: Tuple[int, int, int] = (960, 640, 25), # width, height, frames
        generator: Optional[torch.Generator] = None,
        timestep_sampler: Optional[TimestepSampler] = None,
        frame_rate: float | Float[torch.Tensor, 'b'] = 25,
        **unused_kwargs,
    ) -> CustomBaseTrainingStrategyContext:
        raise NotImplementedError()


# region: Pi3 probe

class LTXVDataset(Dataset):
    @property
    def discrete_reference_indices(self):
        if isinstance(self._discrete_reference_indices, float):
            return random.random() < self._discrete_reference_indices
        return self._discrete_reference_indices

    @property
    def discrete_condition_indices(self):
        if isinstance(self._discrete_condition_indices, float):
            return random.random() < self._discrete_condition_indices
        return self._discrete_condition_indices

    def random_slice(self, length: int, total_length: int):
        slice_start = random.randint(0, total_length - length)
        slice_stop = slice_start + self.num_frames
        return slice(slice_start, slice_stop)

    def random_indices(self, length: int, total_length: int) -> List[int]:
        return torch.randperm(total_length)[:length].tolist()

    def __init__(
        self, data_root: str | Path, num_frames: int, num_cond_frames: int, resolution: Tuple[int, int] = (640, 960), *,
        discrete_reference_indices: bool | float = False, discrete_condition_indices: bool | float = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        assert self.data_root.is_dir(), f"Data root {data_root} is not a directory."

        import json
        with open(self.data_root / 'dataset.json') as f:
            self.items = json.load(f)

        self.num_frames = num_frames
        self.num_cond_frames = num_cond_frames
        self.resolution = resolution

        self._discrete_reference_indices = discrete_reference_indices
        self._discrete_condition_indices = discrete_condition_indices

        if (self.data_root / 'default_prompt.pt').exists():
            self.default_prompt = torch.load(self.data_root / 'default_prompt.pt')
        else:
            self.default_prompt = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> Any:
        item = self.items[index % len(self)]
        prompt = torch.load((self.data_root / '.precomputed/conditions' / item['media_path']).with_suffix('.pt'))

        target_decoder = VideoDecoder(self.data_root / item['media_path'], dimension_order='NCHW')
        total_frames = target_decoder.metadata.num_frames or 0
        if total_frames < self.num_frames:
            print(f"Error: too few frames ({total_frames}) in {item}.")
            self.items.pop(index)
            return self[index % len(self)]
        if self.discrete_reference_indices:
            ref_selection = self.random_indices(self.num_cond_frames, total_frames)
            # reference_pixels = ref_pixels_decoder.get_frames_at(indices=ref_selection).data
            # reference_depths = ref_depths_decoder.get_frames_at(indices=ref_selection).data
            target_pixels = target_decoder.get_frames_at(indices=ref_selection).data
        else:
            ref_selection = self.random_slice(self.num_frames, total_frames)
            # reference_pixels = ref_pixels_decoder[ref_selection]
            # reference_depths = ref_depths_decoder[ref_selection]
            target_pixels = target_decoder[ref_selection]
        # reference_extrinsics = extrinsics[ref_selection]
        # reference_intrinsics = intrinsics[ref_selection]

        return {
            'fps': target_decoder.metadata.average_fps or 25,
            'prompt_embeds': prompt['prompt_embeds'],
            'prompt_attention_mask': prompt['prompt_attention_mask'],
            'target_pixels': F.interpolate(target_pixels.float() / 255, size=self.resolution, mode='bilinear', align_corners=False), # t c h w

            # 'reference_pixels': F.interpolate(reference_pixels.float() / 255, size=self.resolution, mode='bilinear', align_corners=False), # t c h w
            # 'reference_depths': F.interpolate(reference_depths.float() / 255, size=self.resolution, mode='bilinear', align_corners=False), # t c h w
            # 'reference_extrinsics': reference_extrinsics, # t 4 4
            # 'reference_intrinsics': reference_intrinsics, # t 3 3

            # 'condition_pixels': F.interpolate(condition_pixels.float() / 255, size=self.resolution, mode='bilinear', align_corners=False), # r c h w
            # 'condition_extrinsics': condition_extrinsics, # r 4 4
            # 'condition_intrinsics': condition_intrinsics, # r 3 3
        }

class CustomPi3EncoderProbeVideoTrainingStrategy(CustomBaseTrainingStrategy):
    @torch.compile
    def pi3_encoder_feature(self, images: Float[torch.Tensor, 'b n c h w']) -> Float[torch.Tensor, 'b n ph pw d']:
        b, n, c, h, w = images.shape # assert c == 3
        images = (images - self._pi3.image_mean) / self._pi3.image_std # type: ignore

        patch_h, patch_w = h // 14, w // 14
        hidden = self._pi3.encoder(rearrange(images, 'b n c h w -> (b n) c h w'), is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden['x_norm_patchtokens']
        return hidden.view(b, n, patch_h, patch_w, -1)

    @torch.compile
    def pi3_decoder_feature(self, encoder_feat: Float[torch.Tensor, 'b n ph pw d']) -> Float[torch.Tensor, 'b n ph pw e']:
        b, n, ph, pw, d = encoder_feat.shape
        h, w = ph * 14, pw * 14
        hidden, pos = self._pi3.decode(rearrange(encoder_feat, 'b n ph pw d -> (b n) (ph pw) d'), n, h, w)
        return hidden[:, self._pi3.patch_start_idx:].view(b, n, ph, pw, -1) # remove register tokens

    @torch.compile
    @torch.no_grad()
    def pi3_feature(self, images: Float[torch.Tensor, 'b n c h w']) -> Float[torch.Tensor, 'b n ph pw 2*d']:
        encoder_feat = self.pi3_encoder_feature(images)
        decoder_feat = self.pi3_decoder_feature(encoder_feat)
        return encoder_feat, decoder_feat

    def __init__(self, conditioning_config: ConditioningConfig, vae: AutoencoderKLLTXVideo, pi3: Any):
        super().__init__(conditioning_config)
        self._vae = vae

        from pi3.models.pi3 import Pi3
        self._pi3 = cast(Pi3, pi3)
        self._pi3_transform = Pi3Config.transform()

        self.override_frame_rate = 10

    def prepare_context(
        self,
        target_pixels: Optional[Float[torch.Tensor, 'b t c h w']] = None,
        first_frame_pixels: Optional[Float[torch.Tensor, 'b f c h w']] = None,
        reference_pixels: Optional[Float[torch.Tensor, 'b r c h w']] = None,
        condition_pixels: Optional[Float[torch.Tensor, 'b n c h w']] = None,
        video_dims: Tuple[int, int, int] = (960, 640, 25), # width, height, frames
        generator: Optional[torch.Generator] = None,
        timestep_sampler: Optional[TimestepSampler] = None,
        frame_rate: float = 25,
        **unused_kwargs,
    ):
        t_compress, s_compress = self._vae.temporal_compression_ratio, self._vae.spatial_compression_ratio

        if target_pixels is not None:
            b, t, c, h, w = target_pixels.shape
        else:
            width, height, frames = video_dims
            b, t, c, h, w = 1, frames, 3, height, width

        latent_frames = (t - 1) // t_compress + 1
        latent_height = h // s_compress
        latent_width = w // s_compress

        if (target_pixels is not None) and (timestep_sampler is not None):
            target_latents = encode_as_video(self._vae, target_pixels) # b c latent_frames latent_height latent_width
            # print(f"{target_latents.std(dim=(1, 3, 4))=}")
            target_condition_mask = torch.zeros((b, 128, latent_frames, latent_height, latent_width), device=target_latents.device, dtype=target_latents.dtype)
        else:
            target_latents = torch.randn((b, 128, latent_frames, latent_height, latent_width), device=self._vae.device, dtype=self._vae.dtype, generator=generator)
            target_condition_mask = torch.zeros((b, 128, latent_frames, latent_height, latent_width), device=target_latents.device, dtype=target_latents.dtype)

        noise = randn_like(target_latents, generator=generator)

        target_video_coords = indices_to_pixel_coords(patch_indices(patchify(target_latents)))
        target_video_coords = rearrange(target_video_coords, 'b f h w c -> b c (f h w)').float()
        target_video_coords[:, 0] = target_video_coords[:, 0] / frame_rate
        video_coords = target_video_coords

        if timestep_sampler is not None:
            assert target_pixels is not None
            # train; target pixels provided; add noise to target latents; may do first frame conditioning; set target_condition_mask[:, :, 0] = 1 if randomly selected
            target_condition_mask[:, :, 0] = (torch.rand((b, 1, 1, 1), device=target_latents.device, generator=generator) < self.conditioning_config.first_frame_conditioning_p).to(target_latents)

            sigmas: Float[torch.Tensor, 'b'] = timestep_sampler.sample_for(tokenize(patchify(target_latents)))
            # sigmas = torch.full_like(sigmas, fill_value=0.01)
            applied_sigmas = torch.min(1 - target_condition_mask, rearrange(sigmas, 'b -> b 1 1 1 1')).to(target_latents) # b 128 f+r+t h w
            hidden_states = torch.lerp(target_latents, noise, applied_sigmas) # as input:hidden_states, (batch_size num_tokens, 128)
            # hidden_states = (1 - applied_sigmas) * target_latents + applied_sigmas * noise # as input:hidden_states, (batch_size num_tokens, 128)

            # sigmas: Float[torch.Tensor, 'b'] = timestep_sampler.sample_for(tokenize(patchify(target_latents)))
            # noisy_latents = (1 - sigmas.view(b, 1, 1, 1, 1)) * target_latents + sigmas.view(b, 1, 1, 1, 1) * noise
            # hidden_states = torch.where(
            #     target_condition_mask > 1 - 1e-3,
            #     target_latents,
            #     noisy_latents,
            # )
            # applied_sigmas = torch.min(1 - target_condition_mask, rearrange(sigmas, 'b -> b 1 1 1 1')).to(target_latents) # b 128 f+r+t h w

            model_pred = noise - target_latents # as prediction target, (batch_size num_tokens, 128)
            # model_pred = target_latents - noise # as prediction target, (batch_size num_tokens, 128)
            # model_pred = noise # as prediction target, (batch_size num_tokens, 128)
            # model_pred = noise - target_latents # as prediction target, (batch_size num_tokens, 128)
            # model_pred = noise - target_latents # as prediction target, (batch_size num_tokens, 128)

            condition_mask = target_condition_mask
        else:
            # inference; target pixels not provided; no additional noise; may do first frame conditioning; set target_condition_mask[:, :, 0] = 1 if first_frame_pixels is provided
            if first_frame_pixels is not None:
                first_frame_latents = encode_as_video(self._vae, first_frame_pixels)
                target_latents[:, :, 0:first_frame_latents.size(2), :, :] = first_frame_latents
                target_condition_mask[:, :, 0:first_frame_latents.size(2), :, :] = 1 # b c latent_frames latent_height latent_width
            applied_sigmas = 1 - target_condition_mask # b 128 f+r+t h w

            hidden_states = target_latents # as input:hidden_states
            model_pred = None # as prediction target
            condition_mask = target_condition_mask

        if target_pixels is not None:
            with torch.no_grad():
                pi3_input_images = self._pi3_transform(target_pixels[:, [max(i-t_compress+1, 0) for i in range(0, t+t_compress-1, t_compress)]]) # b latent_frames c h w
                pi3_input_images = F.interpolate(rearrange(pi3_input_images, 'b t c h w -> (b t) c h w'), size=(latent_height*14, latent_width*14), mode='bicubic', align_corners=False, antialias=True)
                encoder_feature, decoder_feature = self.pi3_feature(rearrange(pi3_input_images.to(self._vae.dtype), '(b t) c h w -> b t c h w', b=b)) # b latent_frames latent_height latent_width c
                latents_add = rearrange(decoder_feature, 'b t h w d -> b (t h w) d')
        else:
            latents_add = None

        # TODO: add condition and reference

        # print(applied_sigmas.mean(dim=(1,3,4))*1000)

        context = {
            'hidden_states': tokenize(patchify(hidden_states)),
            'timestep': tokenize(patchify(applied_sigmas)).mean(-1) * 1000,
            'video_coords': video_coords,
            'latents_add': latents_add,

            'condition_mask': tokenize(patchify(condition_mask)),
            'model_pred': tokenize(patchify(model_pred)), # not the input to the model, but the training target
            'noise': tokenize(patchify(noise)),
        }
        # print(f"{describe(context)=}")
        return context

class CustomPi3EncoderProbeTrainer(LtxvTrainer):
    def _prepare_models_for_training(self) -> None:
        super()._prepare_models_for_training()
        self._vae = self._vae.to(self._accelerator.device)

    def _compile_transformer(self) -> None:
        """Compile the transformer model with Torch Inductor."""
        super()._compile_transformer()
        torch._dynamo.config.capture_scalar_outputs = True
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

        if self._transformer.latents_add_proj_in is not None:
            self._transformer.latents_add_proj_in.requires_grad_(True)

        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        # self._trainable_params = [p for p in self._transformer.latents_add_proj_in.parameters() if p.requires_grad]
        logger.debug(f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            width, height, frames = self._config.validation.video_dims
            self._dataset = CustomDL3DV10KDataset(
                # self._config.data.preprocessed_data_root, num_frames=25, num_cond_frames=8, resolution=(640, 960),
                self._config.data.preprocessed_data_root, num_frames=frames, num_cond_frames=8, resolution=(height, width),
                discrete_reference_indices=False, discrete_condition_indices=True,
            )
            # self._dataset = LTXVDataset(
            #     self._config.data.preprocessed_data_root, num_frames=frames, num_cond_frames=8, resolution=(height, width),
            #     discrete_reference_indices=False, discrete_condition_indices=True,
            # )
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

        # print(f"{loss.item()=:.4f} {training_batch['timestep'].mean().item()=:.3f}")
        # print(f"{training_batch['model_pred'].reshape(1, -1, 20*30, 128).std(dim=(2, 3))=}")
        # print(f"{training_batch['hidden_states'].reshape(1, -1, 20*30, 128).std(dim=(2, 3))=}")
        # hidden_states = training_batch['hidden_states'].reshape(1, -1, 20*30, 128)[:, 1:].reshape(-1).float()
        # noise = training_batch['noise'].reshape(1, -1, 20*30, 128)[:, 1:].reshape(-1).float()
        # X = torch.stack([hidden_states, noise], dim=1)
        # w = torch.linalg.lstsq(X, model_pred.reshape(1, -1, 20*30, 128)[:, 1:].reshape(-1).float()).solution
        # print(f"{w=}")

        # def _compute_loss(target):
        #     loss = (model_pred - target).pow(2)
        #     # Create loss mask: exclude conditioning tokens
        #     loss_mask = (1 - training_batch['condition_mask']).float()
        #     # Apply original loss computation pattern
        #     loss = loss.mul(loss_mask).div(loss_mask.mean() + 1e-3)
        #     return loss.mean()
        # print(f"="*20)
        # print(f"{loss.item()=:.4f} {training_batch['timestep'].mean().item()=:.3f}")
        # print(f"{_compute_loss(training_batch['model_pred']).item():.4f}")
        # print(f"{_compute_loss(training_batch['hidden_states']).item():.4f}")
        # print(f"{_compute_loss(training_batch['noise']).item():.4f}")
        # print(f"{_compute_loss(training_batch['noise']-training_batch['hidden_states']).item():.4f}")

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
                context_kwargs = {
                    'video_dims': self._config.validation.video_dims,
                    'generator': generator,
                    'timestep_sampler': None,
                    'frame_rate': self._training_strategy._override_frame_rate or self._training_strategy._default_frame_rate,
                }

                if sample.first_frame is not None:
                    if isinstance(sample.first_frame, str):
                        first_frames = torch.as_tensor(iio.imread(sample.first_frame)).to(self._accelerator.device).float()[None].permute(0,3,1,2) / 255
                    else:
                        video_decoder = VideoDecoder(sample.first_frame.path, dimension_order='NCHW')
                        first_frames = video_decoder[sample.first_frame.index:sample.first_frame.index+1].to(self._accelerator.device).float() / 255
                    first_frames = F.interpolate(first_frames, size=(height, width), mode='bicubic', align_corners=False, antialias=True).to(self._vae.dtype) # t c h w
                    context_kwargs['first_frame_pixels'] = first_frames[None, 0:frames]

                if sample.target_video is not None:
                    target_video_path = sample.target_video if isinstance(sample.target_video, str) else sample.target_video.path
                    target_video_slice = slice(None, None, None) if isinstance(sample.target_video, str) else slice(sample.target_video.start, sample.target_video.end, sample.target_video.step)

                    target_video_decoder = VideoDecoder(target_video_path, dimension_order='NCHW')
                    target_pixels = target_video_decoder[target_video_slice].to(self._accelerator.device).float() / 255
                    target_pixels = F.interpolate(target_pixels, size=(height, width), mode='bicubic', align_corners=False, antialias=True).to(self._vae.dtype) # t c h w
                    context_kwargs['target_pixels'] = target_pixels[None, 0:frames]
                
                if sample.reference_video is not None:
                    reference_video_path = sample.reference_video if isinstance(sample.reference_video, str) else sample.reference_video.path
                    reference_video_slice = slice(None, None, None) if isinstance(sample.reference_video, str) else slice(sample.reference_video.start, sample.reference_video.end, sample.reference_video.step)

                    reference_video_decoder = VideoDecoder(reference_video_path, dimension_order='NCHW')
                    reference_pixels = reference_video_decoder[reference_video_slice].to(self._accelerator.device).float() / 255
                    reference_pixels = F.interpolate(reference_pixels, size=(height, width), mode='bicubic', align_corners=False, antialias=True).to(self._vae.dtype)
                    context_kwargs['reference_pixels'] = reference_pixels[None, 0:frames]

                context = self._training_strategy.prepare_context(**context_kwargs)
                context = {k: (v.to(cast(torch.dtype, self._transformer.dtype)) if isinstance(v, torch.Tensor) else v) for k, v in context.items()}

            with torch.amp.autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                latents = pipeline.latents_sample_loop(
                    latents=context['hidden_states'],
                    latent_coords=context['video_coords'],
                    prompt_embeds=prompt_embeds['prompt_embeds'],
                    prompt_attention_mask=prompt_embeds['prompt_attention_mask'],
                    negative_prompt_embeds=negative_prompts['prompt_embeds'],
                    negative_prompt_attention_mask=negative_prompts['prompt_attention_mask'],
                    num_inference_steps=self._config.validation.inference_steps,
                    conditioning=(context['hidden_states'], context['condition_mask']) if 'condition_mask' in context else None,
                    latents_add=context.get('latents_add', None),
                )
                latents = rearrange(latents, 'b (fp hp wp) (c pf ph pw) -> b c (fp pf) (hp ph) (wp pw)', fp=latent_frames, hp=latent_height, wp=latent_width, pf=1, ph=1, pw=1)
                videos = pipeline.sample_videos(latents, device=self._accelerator.device)

                # with torch.no_grad():
                #     # one-step denoising
                #     # one_step = pipeline.latents_sample_loop(
                #     #     latents=context['hidden_states'],
                #     #     latent_coords=context['video_coords'],
                #     #     prompt_embeds=prompt_embeds['prompt_embeds'],
                #     #     prompt_attention_mask=prompt_embeds['prompt_attention_mask'],
                #     #     negative_prompt_embeds=negative_prompts['prompt_embeds'],
                #     #     negative_prompt_attention_mask=negative_prompts['prompt_attention_mask'],
                #     #     num_inference_steps=1,
                #     #     conditioning=(context['hidden_states'], context['condition_mask']) if 'condition_mask' in context else None,
                #     #     latents_add=context.get('latents_add', None),
                #     # )
                #     # one_step = rearrange(one_step, 'b (fp hp wp) (c pf ph pw) -> b c (fp pf) (hp ph) (wp pw)', fp=latent_frames, hp=latent_height, wp=latent_width, pf=1, ph=1, pw=1)
                #     model_pred = cast(TokenizedLatents, self._transformer(
                #         hidden_states=context['hidden_states'],
                #         encoder_hidden_states=prompt_embeds['prompt_embeds'],
                #         encoder_attention_mask=prompt_embeds['prompt_attention_mask'],
                #         video_coords=context['video_coords'],
                #         timestep=context['timestep'],
                #         latents_add=context.get('latents_add', None),
                #         return_dict=True
                #     ).sample) # model_pred should be equal to noise - target_latents; target_latents = noise - model_pred
                #     one_step = context['hidden_states'] + model_pred
                #     one_step = rearrange(one_step, 'b (fp hp wp) (c pf ph pw) -> b c (fp pf) (hp ph) (wp pw)', fp=latent_frames, hp=latent_height, wp=latent_width, pf=1, ph=1, pw=1)
                # videos = pipeline.sample_videos(one_step, device=self._accelerator.device)

            for video in videos:
                video_path = output_dir / f"step_{self._global_step:06d}_{i}.mp4"
                torchvision.io.write_video(str(video_path), video.permute(1,2,3,0).clamp(0,1).cpu()*255, fps=24, options={'crf': '20'})
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

# endregion

# region: advanced trainer

class CustomDenseDL3DV10KDataset(Dataset):
    def __init__(
        self, data_root: str | Path,
        num_frames: int, num_cond_frames: int, resolution: Tuple[int, int] = (640, 960), *,
        discrete_reference_indices: bool | float = False, discrete_condition_indices: bool | float = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        assert self.data_root.is_dir(), f"Data root {data_root} is not a directory."
        try:
            index = read_json(self.data_root / 'index.json')
        except:
            index = {
                d: read_json(d / 'metrics.json')
                for d in tqdm(list(self.data_root.iterdir()))
                if (
                    (d / 'metrics.json').is_file() and
                    (d / '.done.gen_dense').is_file() and
                    (d / '.done.gen_sparse.v3').is_file() and
                    (d / 'prompt.pt').is_file()
                )
            }
            index = {
                str(d.relative_to(self.data_root)): {
                    'psnr': v['psnr'],
                    'ssim': v['ssim'],
                    'lpips': v['lpips'],
                }
                for d, v in index.items()
                if v.get('psnr', 0) > 29.0
            }
            save_json(index, self.data_root / 'index.json')

        # self.items = [d for d in self.data_root.iterdir() if (d / '.done.dense').exists()]
        # self.items = [d for d in self.items if (d / 'prompt.pt').is_file()]
        self.items = [self.data_root / k for k in index.keys()]

        self.num_frames = num_frames
        self.num_cond_frames = num_cond_frames
        self.resolution = resolution

        self._discrete_reference_indices = discrete_reference_indices
        self._discrete_condition_indices = discrete_condition_indices

        if (self.data_root / 'default_prompt.pt').exists():
            self.default_prompt = torch.load(self.data_root / 'default_prompt.pt')
        else:
            self.default_prompt = None

    def __len__(self):
        return len(self.items)

    @property
    def discrete_reference_indices(self):
        if isinstance(self._discrete_reference_indices, float):
            return random.random() < self._discrete_reference_indices
        return self._discrete_reference_indices

    @property
    def discrete_condition_indices(self):
        if isinstance(self._discrete_condition_indices, float):
            return random.random() < self._discrete_condition_indices
        return self._discrete_condition_indices

    def random_slice(self, length: int, total_length: int):
        slice_start = random.randint(0, total_length - length * 2)
        slice_stop = slice_start + self.num_frames * 2
        return slice(slice_start, slice_stop, 2)

    def random_indices(self, length: int, total_length: int) -> List[int]:
        return torch.randperm(total_length)[:length].tolist()

    def __getitem__(self, index: int):
        item_dir = self.items[index % len(self)]
        # camera_params = torch.load(item_dir / 'camera_params.pth')
        # extrinsics: Float[torch.Tensor, 'n 4 4'] = camera_params['extrinsics'] # n 4 4, c2w
        # intrinsics: Float[torch.Tensor, 'n 3 3'] = camera_params['intrinsics'] # n 3 3
        # total_frames = intrinsics.size(0)

        ground_truth_decoder = VideoDecoder(item_dir / 'dense_pixels.mp4')
        ref_pixels_decoder = VideoDecoder(item_dir / 'sparse_pixels.mp4')
        # ref_depths_decoder = VideoDecoder(item_dir / 'render_depths.mp4')
        total_frames = min(not_none(ground_truth_decoder.metadata.num_frames), not_none(ref_pixels_decoder.metadata.num_frames))

        if total_frames < self.num_frames * 2:
            print(f"Error: too few frames ({total_frames}) in {item_dir}.")
            self.items.pop(index)
            return self[index % len(self)]

        if self.discrete_reference_indices:
            ref_selection = self.random_indices(self.num_cond_frames, total_frames)
            reference_pixels = ref_pixels_decoder.get_frames_at(indices=ref_selection).data
            # reference_depths = ref_depths_decoder.get_frames_at(indices=ref_selection).data
            target_pixels = ground_truth_decoder.get_frames_at(indices=ref_selection).data
        else:
            ref_selection = self.random_slice(self.num_frames, total_frames)
            reference_pixels = ref_pixels_decoder[ref_selection]
            # reference_depths = ref_depths_decoder[ref_selection]
            target_pixels = ground_truth_decoder[ref_selection]
        # reference_extrinsics = extrinsics[ref_selection]
        # reference_intrinsics = intrinsics[ref_selection]

        if self.discrete_condition_indices:
            cond_selection = self.random_indices(self.num_cond_frames, total_frames)
            condition_pixels = ground_truth_decoder.get_frames_at(indices=cond_selection).data
        else:
            cond_selection = self.random_slice(self.num_cond_frames, total_frames)
            condition_pixels = ground_truth_decoder[cond_selection]
        # condition_extrinsics = extrinsics[cond_selection]
        # condition_intrinsics = intrinsics[cond_selection]

        if (item_dir / 'prompt.pt').is_file():
            prompt = torch.load(item_dir / 'prompt.pt')
        else:
            assert self.default_prompt is not None, "Default prompt not found in data root."
            prompt = self.default_prompt

        return {
            'fps': ground_truth_decoder.metadata.average_fps or 30,
            'prompt_embeds': prompt['prompt_embeds'],
            'prompt_attention_mask': prompt['prompt_attention_mask'],
            'target_pixels': F.interpolate(target_pixels.float() / 255, size=self.resolution, mode='bilinear', align_corners=False), # t c h w

            'reference_pixels': F.interpolate(reference_pixels.float() / 255, size=self.resolution, mode='bilinear', align_corners=False), # t c h w
            # 'reference_depths': F.interpolate(reference_depths.float() / 255, size=self.resolution, mode='bilinear', align_corners=False), # t c h w
            # 'reference_extrinsics': reference_extrinsics, # t 4 4
            # 'reference_intrinsics': reference_intrinsics, # t 3 3

            'condition_pixels': F.interpolate(condition_pixels.float() / 255, size=self.resolution, mode='bilinear', align_corners=False), # r c h w
            # 'condition_extrinsics': condition_extrinsics, # r 4 4
            # 'condition_intrinsics': condition_intrinsics, # r 3 3
        }

class AdvancedModelConfig(ModelConfig):
    transformer_extras: Dict[str, Any] = Field(default_factory=dict)

class AdvancedTrainerConfig(LtxvTrainerConfig):
    model: AdvancedModelConfig = Field(default_factory=AdvancedModelConfig)
    validation: CustomValidationConfig = Field(default_factory=CustomValidationConfig)

    dinov3: DINOv3Config = Field(default_factory=DINOv3Config)
    pi3: Pi3Config = Field(default_factory=Pi3Config)

    @model_validator(mode="after")
    def validate_conditioning_compatibility(self) -> Self:
        return self

class AdvancedVideoTrainingStrategy(CustomBaseTrainingStrategy):
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
    @torch.no_grad()
    def pi3_feature(self, images: Float[torch.Tensor, 'b n c h w']) -> Float[torch.Tensor, 'b n ph pw 3*d']:
        encoder_feat = self.pi3_encoder_feature(images)
        decoder_feat = self.pi3_decoder_feature(encoder_feat)
        # return encoder_feat, decoder_feat
        # return torch.cat([encoder_feat, decoder_feat], dim=-1)
        return decoder_feat

    def __init__(self, conditioning_config: ConditioningConfig, vae: AutoencoderKLLTXVideo, pi3: Any):
        super().__init__(conditioning_config)
        self._vae = vae

        from pi3.models.pi3 import Pi3
        self._pi3 = cast(Pi3, pi3)
        self._pi3_transform = Pi3Config.transform()

        self.override_frame_rate = 10

    def prepare_context(
        self,
        target_pixels: Optional[Float[torch.Tensor, 'b t c h w']] = None,
        first_frame_pixels: Optional[Float[torch.Tensor, 'b f c h w']] = None,
        reference_pixels: Optional[Float[torch.Tensor, 'b r c h w']] = None,
        condition_pixels: Optional[Float[torch.Tensor, 'b n c h w']] = None,
        video_dims: Tuple[int, int, int] = (960, 640, 25), # width, height, frames
        generator: Optional[torch.Generator] = None,
        timestep_sampler: Optional[TimestepSampler] = None,
        frame_rate: float = 25,

        # _is_low_fps = True # 假设 reference, target 是从高帧率视频中降采样的视频, 应当插帧; 对于推理, 应当向 reference 中插帧;
        _is_low_fps = False, # 假设 reference, target 是全帧率视频, 输出视频应当与之一一对应; 对于训练, 应当从 reference 中抽帧;
        # _is_low_fps = None, # 从 timestep_sampler 中推断

        **unused_kwargs,
    ):
        t_compress, s_compress = self._vae.temporal_compression_ratio, self._vae.spatial_compression_ratio

        _is_low_fps = False # override this for debugging
        if _is_low_fps is None:
            _is_low_fps = (timestep_sampler is None) # 推理时 reference 是低帧率视频; 训练时 reference 是全帧率视频
        
        assert reference_pixels is not None, "reference_pixels must be provided"
        assert condition_pixels is not None, "condition_pixels must be provided"

        # sizes and dims
        if target_pixels is not None:
            b, t, c, h, w = target_pixels.shape
        else:
            width, height, frames = video_dims
            b, t, c, h, w = 1, frames, 3, height, width
        latent_frames = (t - 1) // t_compress + 1
        latent_height = h // s_compress
        latent_width = w // s_compress

        if (target_pixels is not None) and (timestep_sampler is not None):
            target_latents = encode_as_video(self._vae, target_pixels) # b c latent_frames latent_height latent_width
            target_condition_mask = torch.zeros((b, 128, latent_frames, latent_height, latent_width), device=target_latents.device, dtype=target_latents.dtype)
        else:
            target_latents = torch.randn((b, 128, latent_frames, latent_height, latent_width), device=self._vae.device, dtype=self._vae.dtype, generator=generator)
            target_condition_mask = torch.zeros((b, 128, latent_frames, latent_height, latent_width), device=target_latents.device, dtype=target_latents.dtype)
        target_video_coords = pixel_to_video_coords(indices_to_pixel_coords(patch_indices(patchify(target_latents))))
        target_video_coords[:, 0] = target_video_coords[:, 0] / frame_rate

        condition_latents = encode_as_images(self._vae, condition_pixels)
        condition_video_coords = pixel_to_video_coords(indices_to_pixel_coords(patch_indices(patchify(condition_latents)), frame_offset=random.randint(-int(20*frame_rate), -int(10*frame_rate))))
        condition_video_coords[:, 0] = condition_video_coords[:, 0] / frame_rate
        condition_sigmas = torch.zeros_like(condition_latents)
        # if _is_low_fps:
        #     low_fps_reference_pixels = reference_pixels
        # else:
        #     low_fps_reference_pixels = reference_pixels[:, [i for i in range(0, reference_pixels.size(1), t_compress)]]
        # reference_latents = encode_as_images(self._vae, low_fps_reference_pixels)
        reference_latents = encode_as_video(self._vae, reference_pixels)
        reference_video_coords = target_video_coords
        reference_sigmas = torch.zeros_like(reference_latents)

        # with torch.no_grad():
        #     pi3_input_images = self._pi3_transform(torch.cat([
        #         condition_pixels, low_fps_reference_pixels
        #     ], dim=1))
        # if target_pixels is not None:
        #     with torch.no_grad():
        #         pi3_input_images = self._pi3_transform(target_pixels[:, [max(i-t_compress+1, 0) for i in range(0, t+t_compress-1, t_compress)]]) # b latent_frames c h w
        #         pi3_input_images = F.interpolate(rearrange(pi3_input_images, 'b t c h w -> (b t) c h w'), size=(latent_height*14, latent_width*14), mode='bicubic', align_corners=False, antialias=True)
        #         encoder_feature, decoder_feature = self.pi3_feature(rearrange(pi3_input_images.to(self._vae.dtype), '(b t) c h w -> b t c h w', b=b)) # b latent_frames latent_height latent_width c
        #         latents_add = rearrange(decoder_feature, 'b t h w d -> b (t h w) d')
        # else:
        #     latents_add = None

        if timestep_sampler is not None:
            assert target_pixels is not None
            # train; target pixels provided; add noise to target latents; may do first frame conditioning; set target_condition_mask[:, :, 0] = 1 if randomly selected
            target_condition_mask[:, :, 0] = (torch.rand((b, 1, 1, 1), device=target_latents.device, generator=generator) < self.conditioning_config.first_frame_conditioning_p).to(target_latents)

            noise = randn_like(target_latents, generator=generator)
            sigmas: Float[torch.Tensor, 'b'] = timestep_sampler.sample_for(tokenize(patchify(target_latents)))
            applied_sigmas = torch.min(1 - target_condition_mask, rearrange(sigmas, 'b -> b 1 1 1 1')).to(target_latents) # b 128 f+r+t h w
            target_hidden_states = torch.lerp(target_latents, noise, applied_sigmas) # as input:hidden_states, (batch_size num_tokens, 128)

            model_pred = noise - target_latents # as prediction target, (batch_size num_tokens, 128)
            condition_mask = target_condition_mask
        else:
            # inference; target pixels not provided; no additional noise; may do first frame conditioning; set target_condition_mask[:, :, 0] = 1 if first_frame_pixels is provided
            if first_frame_pixels is not None:
                first_frame_latents = encode_as_video(self._vae, first_frame_pixels)
                target_latents[:, :, 0:first_frame_latents.size(2), :, :] = first_frame_latents
                target_condition_mask[:, :, 0:first_frame_latents.size(2), :, :] = 1 # b c latent_frames latent_height latent_width
            applied_sigmas = 1 - target_condition_mask # b 128 f+r+t h w

            target_hidden_states = target_latents # as input:hidden_states
            model_pred = None # as prediction target
            condition_mask = target_condition_mask

        # assemble context
        hidden_states = torch.cat([condition_latents, reference_latents, target_hidden_states], dim=2) # b c latent_frames+r h w
        video_coords = torch.cat([condition_video_coords, reference_video_coords, target_video_coords], dim=-1) # b c latent_frames*latent_height*latent_width*2
        applied_sigmas = torch.cat([condition_sigmas, reference_sigmas, applied_sigmas], dim=2)
        condition_mask = torch.cat([1 - condition_sigmas, 1 - reference_sigmas, condition_mask], dim=2)
        if model_pred is not None:
            model_pred = torch.cat([torch.zeros_like(condition_latents), torch.zeros_like(reference_latents), model_pred], dim=2)

        with torch.no_grad():
            # pi3_input_images = self._pi3_transform(torch.cat([condition_pixels, reference_pixels[:, [max(i-t_compress+1, 0) for i in range(0, t+t_compress-1, t_compress)]]], dim=1))
            # pi3_input_images = F.interpolate(rearrange(pi3_input_images, 'b t c h w -> (b t) c h w'), size=(latent_height*14, latent_width*14), mode='bicubic', align_corners=False, antialias=True)
            # feature = self.pi3_feature(rearrange(pi3_input_images.to(self._vae.dtype), '(b t) c h w -> b t c h w', b=b)) # b (r+t) latent_height latent_width c
            # condition_feature = feature[:, 0:condition_latents.size(2)]
            # reference_feature = feature[:, condition_latents.size(2):condition_latents.size(2)+reference_latents.size(2)]
            # latents_add = rearrange(torch.cat([
            #     condition_feature, reference_feature, reference_feature
            # ], dim=1), 'b t h w d -> b (t h w) d')
            latents_add = None

        context = {
            'hidden_states': tokenize(patchify(hidden_states)),
            'timestep': tokenize(patchify(applied_sigmas)).mean(-1) * 1000,
            'video_coords': video_coords,
            'latents_add': latents_add,

            'condition_mask': tokenize(patchify(condition_mask)),
            'model_pred': tokenize(patchify(model_pred)), # not the input to the model, but the training target
            # 'noise': tokenize(patchify(noise)),
        }
        return context

class AdvancedTrainer(LtxvTrainer):
    _config: AdvancedTrainerConfig

    def _prepare_models_for_training(self) -> None:
        super()._prepare_models_for_training()
        self._vae = self._vae.to(self._accelerator.device)

    def _load_models(self) -> None:
        """Load the LTXV model components."""
        # Load all model components using the new loader
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        # Prepare components with accelerator
        self._scheduler = load_scheduler()
        self._tokenizer = load_tokenizer()
        self._text_encoder = load_text_encoder()
        self._vae = load_vae(self._config.model.model_source, dtype=torch.bfloat16)
        self._transformer = load_ltxv_xfmr_2b_manually(LTXVideoTransformer3DModel, torch_dtype=transformer_dtype, **self._config.model.transformer_extras)
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

    def _save_checkpoint(self) -> Path:
        """Save the model weights."""
        if self._config.model.training_mode == "full":
            return super()._save_checkpoint()
        elif self._config.model.training_mode == "lora":
            saved_weights_path = super()._save_checkpoint()
            lora_state_dict = load_file(saved_weights_path, device="cpu")

            # insert extra modules' state dict of self._transformer
            lora_state_dict.update({
                k: v.cpu()
                for k, v in self._transformer.state_dict().items()
                if any(k.startswith(prefix) for prefix in getattr(self._transformer, 'extra_modules', set()))
            })

            # write back
            save_file(lora_state_dict, saved_weights_path)
            return saved_weights_path
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

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

        # if self._transformer.latents_add_proj_in is not None:
        #     self._transformer.latents_add_proj_in.requires_grad_(True)
        for prefix in getattr(self._transformer, 'extra_modules', set()):
            module = getattr(self._transformer, prefix, None)
            if module is not None:
                module.requires_grad_(True)

        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        logger.debug(f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def __init__(self, trainer_config: AdvancedTrainerConfig, training_strategy_cls = AdvancedVideoTrainingStrategy) -> None:
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
        self._training_strategy = training_strategy_cls(self._config.conditioning, self._vae, self._pi3)

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            width, height, frames = self._config.validation.video_dims
            # self._dataset = CustomDL3DV10KDataset(
            #     # self._config.data.preprocessed_data_root, num_frames=25, num_cond_frames=8, resolution=(640, 960),
            #     self._config.data.preprocessed_data_root, num_frames=frames, num_cond_frames=8, resolution=(height, width),
            #     discrete_reference_indices=False, discrete_condition_indices=True,
            # )
            # self._dataset = LTXVDataset(
            #     self._config.data.preprocessed_data_root, num_frames=frames, num_cond_frames=8, resolution=(height, width),
            #     discrete_reference_indices=False, discrete_condition_indices=True,
            # )
            self._dataset = CustomDenseDL3DV10KDataset(
                self._config.data.preprocessed_data_root, num_frames=frames, num_cond_frames=8, resolution=(height, width),
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
                context_kwargs = {
                    'video_dims': self._config.validation.video_dims,
                    'generator': generator,
                    'timestep_sampler': None,
                    'frame_rate': self._training_strategy._override_frame_rate or self._training_strategy._default_frame_rate,
                }

                if sample.first_frame is not None:
                    if isinstance(sample.first_frame, str):
                        first_frames = torch.as_tensor(iio.imread(sample.first_frame)).to(self._accelerator.device).float()[None].permute(0,3,1,2) / 255
                    else:
                        video_decoder = VideoDecoder(sample.first_frame.path, dimension_order='NCHW')
                        first_frames = video_decoder[sample.first_frame.index:sample.first_frame.index+1].to(self._accelerator.device).float() / 255
                    first_frames = F.interpolate(first_frames[0:1], size=(height, width), mode='bicubic', align_corners=False, antialias=True).to(self._vae.dtype) # t c h w
                    context_kwargs['first_frame_pixels'] = first_frames[None]

                if sample.target_video is not None:
                    target_video_path = sample.target_video if isinstance(sample.target_video, str) else sample.target_video.path
                    target_video_slice = slice(None, None, None) if isinstance(sample.target_video, str) else slice(sample.target_video.start, sample.target_video.end, sample.target_video.step)

                    target_video_decoder = VideoDecoder(target_video_path, dimension_order='NCHW')
                    target_pixels = target_video_decoder[target_video_slice][0:frames].to(self._accelerator.device).float() / 255
                    target_pixels = F.interpolate(target_pixels, size=(height, width), mode='bicubic', align_corners=False, antialias=True).to(self._vae.dtype) # t c h w
                    context_kwargs['target_pixels'] = target_pixels[None]
                
                if sample.reference_video is not None:
                    reference_video_path = sample.reference_video if isinstance(sample.reference_video, str) else sample.reference_video.path
                    reference_video_slice = slice(None, None, None) if isinstance(sample.reference_video, str) else slice(sample.reference_video.start, sample.reference_video.end, sample.reference_video.step)

                    reference_video_decoder = VideoDecoder(reference_video_path, dimension_order='NCHW')
                    reference_pixels = reference_video_decoder[reference_video_slice][0:frames].to(self._accelerator.device).float() / 255
                    reference_pixels = F.interpolate(reference_pixels, size=(height, width), mode='bicubic', align_corners=False, antialias=True).to(self._vae.dtype)
                    context_kwargs['reference_pixels'] = reference_pixels[None]

                if sample.condition_video is not None:
                    condition_video_path = sample.condition_video if isinstance(sample.condition_video, str) else sample.condition_video.path
                    condition_video_decoder = VideoDecoder(condition_video_path, dimension_order='NCHW')

                    if isinstance(sample.condition_video, str):
                        condition_pixels = condition_video_decoder[0+frames:32+frames:4].to(self._accelerator.device).float() / 255
                    else:
                        condition_video_slice = slice(sample.condition_video.start, sample.condition_video.end, sample.condition_video.step)
                        condition_pixels = condition_video_decoder[condition_video_slice][0:8].to(self._accelerator.device).float() / 255
                    condition_pixels = F.interpolate(condition_pixels, size=(height, width), mode='bicubic', align_corners=False, antialias=True).to(self._vae.dtype)
                    context_kwargs['condition_pixels'] = condition_pixels[None]

                context = self._training_strategy.prepare_context(**context_kwargs)
                context = {k: (v.to(cast(torch.dtype, self._transformer.dtype)) if isinstance(v, torch.Tensor) else v) for k, v in context.items()}

            with torch.amp.autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                latents = pipeline.latents_sample_loop(
                    latents=context['hidden_states'],
                    latent_coords=context['video_coords'],
                    prompt_embeds=prompt_embeds['prompt_embeds'],
                    prompt_attention_mask=prompt_embeds['prompt_attention_mask'],
                    negative_prompt_embeds=negative_prompts['prompt_embeds'],
                    negative_prompt_attention_mask=negative_prompts['prompt_attention_mask'],
                    num_inference_steps=self._config.validation.inference_steps,
                    conditioning=(context['hidden_states'], context['condition_mask']) if 'condition_mask' in context else None,
                    latents_add=context.get('latents_add', None),
                )
                latents = rearrange(latents[:, -latent_frames*latent_height*latent_width:, :], 'b (fp hp wp) (c pf ph pw) -> b c (fp pf) (hp ph) (wp pw)', fp=latent_frames, hp=latent_height, wp=latent_width, pf=1, ph=1, pw=1)
                videos = pipeline.sample_videos(latents, device=self._accelerator.device)
            for video in videos:
                video_path = output_dir / f"step_{self._global_step:06d}_{i}.mp4"
                torchvision.io.write_video(str(video_path), video.permute(1,2,3,0).clamp(0,1).cpu()*255, fps=24, options={'crf': '20'})
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

# endregion

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
            trainer_config = AdvancedTrainerConfig(**config_data)
        except Exception as e:
            typer.echo(f"Error: Invalid configuration data: {e}")
            raise typer.Exit(code=1) from e
        
        import shutil
        Path(trainer_config.output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(__file__, Path(trainer_config.output_dir) / 'trainer.py')

        # Initialize the training process
        # trainer = CustomTrainer(trainer_config)
        # trainer = CustomPi3EncoderProbeTrainer(trainer_config)
        trainer = AdvancedTrainer(trainer_config)
        setattr(trainer, 'config_path', config_path)
        # print(f"{trainer._scheduler.sigmas=}") # 1->0单调减少
        # print(f"{trainer._scheduler.config.get('stochastic_sampling', None)}") # False

        # sample_progress = Progress(
        #     TextColumn("Sampling validation videos"),
        #     MofNCompleteColumn(),
        #     BarColumn(bar_width=40, style="blue"),
        #     TimeElapsedColumn(),
        #     TextColumn("ETA:"),
        #     TimeRemainingColumn(compact=True),
        # )
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
