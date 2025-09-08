import torch
import torchvision
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import copy

from einops import *
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder
from typing import *
from jaxtyping import Float, Shaped, Int, Integer, Bool
from pydantic import computed_field
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Group,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from dinov3.models.vision_transformer import DinoVisionTransformer

from gshub.utils import not_none_or_dotenv, describe

from ltxv_trainer import logger
from ltxv_trainer.trainer import LtxvTrainer
from ltxv_trainer.config import *
from ltxv_trainer.custom.transformer_ltx import LTXVideoTransformer3DModel
from ltxv_trainer.custom.ltxv_pipeline import LTXReferenceConditionPipeline
from ltxv_trainer.training_strategies import get_training_strategy, TrainingStrategy, TrainingBatch, DEFAULT_FPS
from ltxv_trainer.timestep_samplers import TimestepSampler
from ltxv_trainer.model_loader import LtxvModelComponents, LtxvModelVersion, load_ltxv_components, load_scheduler, load_text_encoder, load_tokenizer, load_vae, load_transformer
from ltxv_trainer.quantization import quantize_model
from ltxv_trainer.ltxv_utils import encode_video, decode_video, get_rope_scale_factors, prepare_video_coordinates
from ltxv_trainer.utils import open_image_as_srgb
from ltxv_trainer.video_utils import read_video
from diffusers.models.autoencoders import AutoencoderKLLTXVideo

MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

# dinov3_vitl16 = cast(DinoVisionTransformer, torch.hub.load('https://github.com/HamiltonHuaji/dinov3', MODEL_DINOV3_VITL, source='github', weights='./checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')).to(device)
dinov3_normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
def dinov3_transform(pixels: Float[torch.Tensor, 't c h w'], patch_size: int = 16) -> Float[torch.Tensor, 't c h w']:
    t, c, h, w = pixels.shape
    if (h % patch_size != 0) or (w % patch_size != 0):
        resize_hw = round(h / patch_size) * patch_size, round(w / patch_size) * patch_size
        pixels = F.interpolate(pixels, size=resize_hw, mode='bicubic', align_corners=False)
    return dinov3_normalize(pixels)

class CustomTrainerConfig(LtxvTrainerConfig):
    dinov3_model_name: str = Field(default=MODEL_DINOV3_VITL)
    dinov3_checkpoint: str | None = Field(default=None)

class CustomDL3DV10KDatasetBatch(TypedDict):
    target_pixels: Float[torch.Tensor, 'b t c h w']
    fps: float | None
    prompt_embeds: Float[torch.Tensor, 'b max_length d']
    prompt_attention_mask: Bool[torch.Tensor, 'b max_length']

    condition_pixels: Float[torch.Tensor, 'b t c h w']
    condition_depths: Float[torch.Tensor, 'b t c h w']
    condition_extrinsics: Float[torch.Tensor, 'b t 4 4']
    condition_intrinsics: Float[torch.Tensor, 'b t 3 3']

    reference_pixels: Float[torch.Tensor, 'b r c h w']
    reference_extrinsics: Float[torch.Tensor, 'b r 4 4']
    reference_intrinsics: Float[torch.Tensor, 'b r 3 3']

class CustomDL3DV10KDataset(Dataset):
    def __init__(self, data_root: str, num_frames: int, num_ref_frames: int):
        super().__init__()
        self.data_root = Path(data_root)
        assert self.data_root.is_dir(), f"Data root {data_root} is not a directory."
        self.items = [d for d in self.data_root.iterdir() if (d / '.done').exists()]
        self.num_frames = num_frames
        self.num_ref_frames = num_ref_frames

        if (self.data_root / 'default_prompt.pt').exists():
            self.default_prompt = torch.load(self.data_root / 'default_prompt.pt')
        else:
            self.default_prompt = None

    def __len__(self):
        return len(self.items)

    def random_slice(self, length: int, total_length: int):
        slice_start = random.randint(0, total_length - 1 - length)
        slice_stop = slice_start + self.num_frames
        return slice(slice_start, slice_stop)

    def random_indices(self, length: int, total_length: int):
        return torch.randperm(total_length)[:length].tolist()

    def __getitem__(self, index: int):
        item_dir = self.items[index]
        camera_params = torch.load(item_dir / 'camera_params.pth')
        extrinsics: Float[torch.Tensor, 'n 4 4'] = camera_params['extrinsics'] # n 4 4, c2w
        intrinsics: Float[torch.Tensor, 'n 3 3'] = camera_params['intrinsics'] # n 3 3
        total_frames = intrinsics.size(0)

        selection = self.random_slice(self.num_frames, total_frames)
        ref_selection = self.random_indices(self.num_frames, total_frames)
        # ref_selection = self.random_indices(self.num_frames, total_frames) # for TODO version

        condition_pixels = VideoDecoder(item_dir / 'render_pixels.mp4')[selection] # t c h w
        condition_depths = VideoDecoder(item_dir / 'render_depths.mp4')[selection] # t c h w
        condition_extrinsics = extrinsics[selection]
        condition_intrinsics = intrinsics[selection]

        ground_truth = VideoDecoder(item_dir / 'ground_truth.mp4')

        reference_pixels = ground_truth.get_frames_at(indices=ref_selection).data
        reference_extrinsics = extrinsics[ref_selection]
        reference_intrinsics = intrinsics[ref_selection]

        assert self.default_prompt is not None, "Default prompt not found in data root."

        return {
            'fps': ground_truth.metadata.average_fps,
            'prompt_embeds': self.default_prompt['prompt_embeds'],
            'prompt_attention_mask': self.default_prompt['prompt_attention_mask'],
            'target_pixels': F.interpolate(ground_truth[selection].float() / 255, size=(640, 960), mode='bilinear', align_corners=False), # t c h w

            'condition_pixels': F.interpolate(condition_pixels.float() / 255, size=(640, 960), mode='bilinear', align_corners=False), # t c h w
            'condition_depths': F.interpolate(condition_depths.float() / 255, size=(640, 960), mode='bilinear', align_corners=False), # t c h w
            'condition_extrinsics': condition_extrinsics, # t 4 4
            'condition_intrinsics': condition_intrinsics, # t 3 3

            'reference_pixels': F.interpolate(reference_pixels.float() / 255, size=(640, 960), mode='bilinear', align_corners=False), # r c h w
            'reference_extrinsics': reference_extrinsics, # r 4 4
            'reference_intrinsics': reference_intrinsics, # r 3 3
        }

class CustomTrainingBatch(BaseModel):
    """Container for prepared training data.

    This model holds all the prepared data needed for a training step,
    organized in a way that's agnostic to the specific training strategy.
    """

    # Core latent data
    latents: Float[torch.Tensor, 'b c t h w']  # The main latent input to the transformer
    targets: Float[torch.Tensor, 'b c t h w']  # The target values for loss computation
    viewmats: Float[torch.Tensor, 'b c t h w']

    # Text conditioning
    prompt_embeds: Float[torch.Tensor, 'b max_length d']  # Text embeddings
    prompt_attention_mask: Bool[torch.Tensor, 'b max_length']  # Attention mask for text

    # Timestep information
    timesteps: torch.Tensor  # Timestep values for the transformer
    sigmas: torch.Tensor  # Noise schedule values

    # Conditioning information
    conditioning_mask: torch.Tensor  # Boolean mask: True = conditioning token, False = target token

    # Video metadata
    num_frames: int  # Number of frames in the video
    height: int  # Height of the video latents
    width: int  # Width of the video latents
    fps: float  # Frames per second

    # Model input parameters
    rope_interpolation_scale: list[float]  # Scaling factors for positional embeddings
    video_coords: torch.Tensor | None = None  # Optional explicit video coordinates

    @computed_field
    @property
    def batch_size(self) -> int:
        """Compute batch size from latents tensor."""
        return self.latents.shape[0]

    @computed_field
    @property
    def sequence_length(self) -> int:
        """Compute sequence length from latents tensor."""
        return self.latents.shape[1]

    model_config = {"arbitrary_types_allowed": True}  # Allow torch.Tensor type

class CustomReferenceVideoTrainingStrategy(TrainingStrategy):
    """Reference video training strategy for IC-LoRA.

    This strategy implements training with reference video conditioning where:
    - Reference latents (clean) are concatenated with target latents (noised)
    - Video coordinates are doubled to handle concatenated sequence
    - Loss is computed only on the target portion (masked loss)
    - Supports first frame conditioning on the target sequence
    """

    def __init__(self, conditioning_config: ConditioningConfig, vae: AutoencoderKLLTXVideo, dinov3: DinoVisionTransformer):
        """Initialize with configurable reference latents directory.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        super().__init__(conditioning_config)
        self._vae = vae
        self._dinov3 = dinov3

    def get_data_sources(self) -> dict[str, str]:
        """This method is useless; Will skip calling this"""
        """IC-LoRA training requires latents, conditions, and reference latents."""
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.conditioning_config.reference_latents_dir: "ref_latents",
        }

    def prepare_batch(self, batch: CustomDL3DV10KDatasetBatch, timestep_sampler: TimestepSampler):
        return self.prepare_batch_baseline(batch, timestep_sampler)

    def prepare_batch_todo(self, batch: CustomDL3DV10KDatasetBatch, timestep_sampler: TimestepSampler) -> CustomTrainingBatch:
        target = encode_video(self._vae, batch['target_pixels'])
        condition = encode_video(self._vae, batch['condition_pixels'])
        condition_depth = encode_video(self._vae, batch['condition_depths'])

        target_latents: Float[torch.Tensor, 'b fhw c'] = target['latents']
        condition_latents: Float[torch.Tensor, 'b fhw c'] = condition['latents']
        condition_depth_latents: Float[torch.Tensor, 'b fhw c'] = condition_depth['latents']
        # Ks and viewmats are shared across target, condition, condition_depth sequences
        condition_Ks: Float[torch.Tensor, 'b f 3 3'] = batch['condition_intrinsics']
        condition_viewmats: Float[torch.Tensor, 'b f 4 4'] = batch['condition_extrinsics'].inverse()

        print(f"{describe(target_latents)=} {describe(condition_latents)=}")

        print(f"{describe(target.get('fps', None))=}")
        latent_frames = target['num_frames'] # pixel frames // patch_size_t(=8)
        latent_height = target['height'] # pixel height // patch_size(=16)
        latent_width = target['width'] # pixel width // patch_size(=16)
        fps = target.get('fps', None) or DEFAULT_FPS
        # Use existing utility function for ROPE scale factors
        rope_scale_factors = get_rope_scale_factors(fps)

        # ref_latents = encode_video(self._vae, batch['reference_pixels'])
        b, r, c, hr, wr = batch['reference_pixels'].shape
        reference_pixels = rearrange(batch['reference_pixels'], 'b r c h w -> (b r) c h w')
        ref_encoding, = self._dinov3.get_intermediate_layers(x=reference_pixels, n=1, reshape=True)
        ref_encoding = rearrange(ref_encoding, '(b r) c ht wt -> b r c ht wt', r=r)
        ref_Ks: Float[torch.Tensor, 'b r 3 3'] = batch['reference_intrinsics']
        ref_viewmats: Float[torch.Tensor, 'b r 4 4'] = batch['reference_extrinsics'].inverse()

        prompt_embeds = batch["prompt_embeds"]
        prompt_attention_mask = batch["prompt_attention_mask"]

        # Create noise only for the target part
        sigmas = timestep_sampler.sample_for(target_latents)
        print(f"{describe(sigmas)=}")

        noise = torch.randn_like(target_latents, device=target_latents.device)
        sigmas = sigmas.view(-1, 1, 1)

        batch_size = target_latents.shape[0]
        cond_seq_len = condition_latents.shape[1]
        target_seq_len = target_latents.shape[1]
        # Reference tokens are always conditioning
        ref_conditioning_mask = torch.ones(batch_size, cond_seq_len, dtype=torch.bool, device=target_latents.device)
        # Target tokens: check for first frame conditioning
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )
        # Combine reference and target conditioning masks
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)
        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)
        # Apply noise only to target part
        noisy_target = (1 - sigmas) * target_latents + sigmas * noise
        # For first frame conditioning in target, use clean latents instead of noisy ones
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)  # (B, target_seq_len, 1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)
        targets = noise - target_latents
        # Concatenate reference and noisy target in the sequence dimension
        # Shape [batch, sequence_length * 2, channels]  # noqa: ERA001
        combined_latents = torch.cat([condition_latents, noisy_target], dim=1)

        # Prepare video coordinates (doubled sequence for concatenation)
        batch_size = combined_latents.shape[0]
        raw_video_coords = prepare_video_coordinates(
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            batch_size=batch_size,
            sequence_multiplier=3,  # IC-LoRA uses doubled sequence (condition latents + condition depth + target)
            device=target_latents.device,
        )
        prescaled_f = raw_video_coords[..., 0] * rope_scale_factors[0]
        prescaled_h = raw_video_coords[..., 1] * rope_scale_factors[1]
        prescaled_w = raw_video_coords[..., 2] * rope_scale_factors[2]
        # Stack to (B, 3, 2*F*H*W) for the transformer's video_coords argument
        video_coords = torch.stack([prescaled_f, prescaled_h, prescaled_w], dim=1)

        return CustomTrainingBatch(
            latents=combined_latents,
            targets=targets,

            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,

            timesteps=timesteps,
            sigmas=sigmas,

            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            rope_interpolation_scale=rope_scale_factors,
            video_coords=video_coords,
        )

    def prepare_batch_baseline(self, batch: CustomDL3DV10KDatasetBatch, timestep_sampler: TimestepSampler) -> TrainingBatch:
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

        # print(f"{describe(batch)=}")

        target = encode_video(self._vae, batch['target_pixels'])
        condition = encode_video(self._vae, batch['condition_pixels'])
        reference = encode_video(self._vae, batch['reference_pixels'])

        target_latents: Float[torch.Tensor, 'b fhw c'] = target['latents']
        condition_latents: Float[torch.Tensor, 'b fhw c'] = condition['latents']
        reference_latents: Float[torch.Tensor, 'b fhw c'] = reference['latents']

        # print(f"{describe(target_latents)=} {describe(condition_latents)=}")
        # print(f"{describe(target.get('fps', None))=}")
        latent_frames = target['num_frames'] # pixel frames // patch_size_t(=8)
        latent_height = target['height'] # pixel height // patch_size(=16)
        latent_width = target['width'] # pixel width // patch_size(=16)
        fps = target.get('fps', None) or DEFAULT_FPS
        # Use existing utility function for ROPE scale factors
        rope_scale_factors = get_rope_scale_factors(fps)

        prompt_embeds = batch["prompt_embeds"]
        prompt_attention_mask = batch["prompt_attention_mask"]

        # Create noise only for the target part
        sigmas = timestep_sampler.sample_for(target_latents)
        # print(f"{describe(sigmas)=}")

        noise = torch.randn_like(target_latents, device=target_latents.device)
        sigmas = sigmas.view(-1, 1, 1)

        batch_size = target_latents.shape[0]
        ref_seq_len = reference_latents.shape[1]
        cond_seq_len = condition_latents.shape[1]
        target_seq_len = target_latents.shape[1]
        # Reference tokens are always conditioning
        reference_mask = torch.ones(batch_size, ref_seq_len, dtype=torch.bool, device=target_latents.device)
        condition_mask = torch.ones(batch_size, cond_seq_len, dtype=torch.bool, device=target_latents.device)
        # Target tokens: check for first frame conditioning
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )
        # Combine reference and target conditioning masks
        conditioning_mask = torch.cat([reference_mask, condition_mask, target_conditioning_mask], dim=1)
        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)
        # Apply noise only to target part
        noisy_target = (1 - sigmas) * target_latents + sigmas * noise
        # For first frame conditioning in target, use clean latents instead of noisy ones
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)  # (B, target_seq_len, 1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)
        targets = noise - target_latents
        # Concatenate reference and noisy target in the sequence dimension
        # Shape [batch, sequence_length * 2, channels]  # noqa: ERA001
        combined_latents = torch.cat([reference_latents, condition_latents, noisy_target], dim=1)

        # Prepare video coordinates (doubled sequence for concatenation)
        batch_size = combined_latents.shape[0]
        raw_video_coords = prepare_video_coordinates(
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            batch_size=batch_size,
            sequence_multiplier=3,  # IC-LoRA uses doubled sequence (reference latents + condition latents + target)
            device=target_latents.device,
        )
        prescaled_f = raw_video_coords[..., 0] * rope_scale_factors[0]
        prescaled_h = raw_video_coords[..., 1] * rope_scale_factors[1]
        prescaled_w = raw_video_coords[..., 2] * rope_scale_factors[2]
        # Stack to (B, 3, 3*F*H*W) for the transformer's video_coords argument
        video_coords = torch.stack([prescaled_f, prescaled_h, prescaled_w], dim=1)

        return TrainingBatch(
            latents=combined_latents,
            targets=targets,

            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,

            timesteps=timesteps,
            sigmas=sigmas,

            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            rope_interpolation_scale=rope_scale_factors,
            video_coords=video_coords,
        )

    def prepare_model_inputs(self, batch: CustomTrainingBatch) -> dict[str, Any]:
        """Prepare inputs for the transformer model.
        Args:
            batch: Prepared training data
        Returns:
            Dictionary of keyword arguments for the transformer forward call
        """
        return {
            "hidden_states": batch.latents,
            "encoder_hidden_states": batch.prompt_embeds,
            "timestep": batch.timesteps,
            "encoder_attention_mask": batch.prompt_attention_mask,
            "num_frames": batch.num_frames,
            "height": batch.height,
            "width": batch.width,
            "rope_interpolation_scale": batch.rope_interpolation_scale,
            "video_coords": batch.video_coords,
            "return_dict": False,
        }

    def prepare_batch_original(self, batch: CustomDL3DV10KDatasetBatch, timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare batch for IC-LoRA training with reference videos."""
        # Get pre-encoded latents
        latents = batch["latents"]
        target_latents = latents["latents"]
        ref_latents = batch["ref_latents"]["latents"]

        # Note: Batch sizes > 1 are partially supported, assuming
        # num_frames, height, width, fps are the same for all batch elements.
        latent_frames = cast(int, latents["num_frames"][0].item())
        latent_height = cast(int, latents["height"][0].item())
        latent_width = cast(int, latents["width"][0].item())

        # Handle FPS with backward compatibility for old preprocessed datasets
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get pre-encoded text conditions
        conditions = batch["conditions"]
        prompt_embeds = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Create noise only for the target part
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents, device=target_latents.device)
        sigmas = sigmas.view(-1, 1, 1)

        # Create conditioning mask
        batch_size = target_latents.shape[0]
        ref_seq_len = ref_latents.shape[1]
        target_seq_len = target_latents.shape[1]

        # Reference tokens are always conditioning
        ref_conditioning_mask = torch.ones(batch_size, ref_seq_len, dtype=torch.bool, device=target_latents.device)

        # Target tokens: check for first frame conditioning
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )

        # Combine reference and target conditioning masks
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        # Apply noise only to target part
        noisy_target = (1 - sigmas) * target_latents + sigmas * noise

        # For first frame conditioning in target, use clean latents instead of noisy ones
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)  # (B, target_seq_len, 1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)

        targets = noise - target_latents

        # Concatenate reference and noisy target in the sequence dimension
        # Shape [batch, sequence_length * 2, channels]  # noqa: ERA001
        combined_latents = torch.cat([ref_latents, noisy_target], dim=1)

        # Use existing utility function for ROPE scale factors
        rope_scale_factors = get_rope_scale_factors(fps)

        # Prepare video coordinates (doubled sequence for concatenation)
        batch_size = combined_latents.shape[0]
        raw_video_coords = prepare_video_coordinates(
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            batch_size=batch_size,
            sequence_multiplier=2,  # IC-LoRA uses doubled sequence (reference + target)
            device=target_latents.device,
        )

        # Apply pre-scaling to raw coordinates.
        # The LTXVideoRotaryPosEmbed expects video_coords to be (B, 3, SeqLen) if provided.
        # It then divides video_coords[:, 0] by base_num_frames, etc.
        # So, the video_coords we pass should be: raw_coord * rope_interpolation_factor
        # (B, 2 * F * H * W)  # noqa: ERA001
        prescaled_f = raw_video_coords[..., 0] * rope_scale_factors[0]
        prescaled_h = raw_video_coords[..., 1] * rope_scale_factors[1]
        prescaled_w = raw_video_coords[..., 2] * rope_scale_factors[2]

        # Stack to (B, 3, 2*F*H*W) for the transformer's video_coords argument
        video_coords = torch.stack([prescaled_f, prescaled_h, prescaled_w], dim=1)

        return TrainingBatch(
            latents=combined_latents,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            rope_interpolation_scale=rope_scale_factors,
            video_coords=video_coords,
        )

    def compute_loss(self, model_pred: torch.Tensor, batch: TrainingBatch) -> torch.Tensor:
        """Compute masked loss only on target portion, excluding conditioning tokens."""
        # Extract target portion from model prediction and conditioning mask
        target_seq_len = batch.targets.shape[1]
        target_pred = model_pred[:, -target_seq_len:]
        target_conditioning_mask = batch.conditioning_mask[:, -target_seq_len:]

        loss = (target_pred - batch.targets).pow(2)

        # Create loss mask: exclude conditioning tokens
        loss_mask = (~target_conditioning_mask.unsqueeze(-1)).float()

        # Apply original loss computation pattern
        loss = loss.mul(loss_mask).div(loss_mask.mean())
        return loss.mean()

class CustomTrainer(LtxvTrainer):
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
        self._training_strategy = CustomReferenceVideoTrainingStrategy(self._config.conditioning, self._vae, self._dinov3)

    def _load_models(self) -> None:
        """Load the LTXV model components, as well as DINOv3 ViT-L."""

        # Load all model components using the new loader
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        model_source=self._config.model.model_source or LtxvModelVersion.latest()
        load_text_encoder_in_8bit=self._config.acceleration.load_text_encoder_in_8bit
        vae_dtype=torch.bfloat16

        components = LtxvModelComponents(
            scheduler=load_scheduler(),
            tokenizer=load_tokenizer(),
            text_encoder=load_text_encoder(load_in_8bit=load_text_encoder_in_8bit),
            vae=load_vae(model_source, dtype=vae_dtype),
            transformer=load_transformer(model_source, dtype=transformer_dtype, transformer_cls=LTXVideoTransformer3DModel),
        )

        # dinov3_checkpoint = not_none_or_dotenv(self._config.dinov3_checkpoint, 'dinov3_vitl16_pretrain_lvd1689m')
        # print(f"{dinov3_checkpoint=}")
        # self._dinov3 = cast(DinoVisionTransformer, torch.hub.load('../dinov3', MODEL_DINOV3_VITL, source='local', weights=dinov3_checkpoint))
        self._dinov3 = nn.Module()

        self._dinov3.requires_grad_(False)

        # Prepare components with accelerator
        self._scheduler = components.scheduler
        self._tokenizer = components.tokenizer
        self._text_encoder = components.text_encoder
        self._vae = components.vae
        self._transformer = components.transformer

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.warning(f"Quantizing model with precision: {self._config.acceleration.quantization}")
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )

        # Freeze all models. We later unfreeze the transformer based on training mode.
        self._text_encoder.requires_grad_(False)
        self._vae.requires_grad_(False)
        self._transformer.requires_grad_(False)

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._training_strategy.get_data_sources()

            self._dataset = CustomDL3DV10KDataset(self._config.data.preprocessed_data_root, num_frames=25, num_ref_frames=25)
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")

        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._config.data.num_dataloader_workers,
            pin_memory=self._config.data.num_dataloader_workers > 0,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

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

    def _training_step(self, batch: CustomDL3DV10KDatasetBatch) -> torch.Tensor:
        """Perform a single training step using the configured strategy."""
        # Use strategy to prepare the training batch
        training_batch = self._training_strategy.prepare_batch(batch, self._timestep_sampler)

        # Use strategy to prepare model inputs
        model_inputs = self._training_strategy.prepare_model_inputs(training_batch)

        # Run transformer forward pass
        model_pred = self._transformer(**model_inputs)[0]

        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(model_pred, training_batch)

        return loss

    @torch.no_grad()
    @torch.compiler.set_stance("force_eager")
    def _sample_videos(self, progress: Progress) -> list[Path] | None:
        """Run validation by generating images from validation prompts."""

        self._vae.to(self._accelerator.device)
        # Model is already in the correct device if loaded in 8-bit.
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to(self._accelerator.device)

        use_images = self._config.validation.images is not None

        pipeline = LTXReferenceConditionPipeline(
            scheduler=copy.deepcopy(self._scheduler),
            vae=self._accelerator.unwrap_model(self._vae),
            text_encoder=self._accelerator.unwrap_model(self._text_encoder),
            tokenizer=self._tokenizer,
            transformer=self._accelerator.unwrap_model(self._transformer),
        )
        pipeline.set_progress_bar_config(disable=True)

        # Create a task in the sampling progress
        task = progress.add_task(
            "sampling",
            total=len(self._config.validation.prompts),
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        video_paths = []
        i = 0
        for j, prompt in enumerate(self._config.validation.prompts):
            generator = torch.Generator(device=self._accelerator.device).manual_seed(self._config.validation.seed)

            # Generate video
            width, height, frames = self._config.validation.video_dims

            pipeline_inputs = {
                "prompt": prompt,
                "negative_prompt": self._config.validation.negative_prompt,
                "width": width,
                "height": height,
                "num_frames": frames,
                "num_inference_steps": self._config.validation.inference_steps,
                "guidance_scale": self._config.validation.guidance_scale,
                "generator": generator,
                "output_reference_comparison": True,
            }

            # Load and add first frame image, if provided
            if use_images:
                image_path = self._config.validation.images[j]
                image = open_image_as_srgb(image_path)
                if image.size != (height, width):
                    # Resize and center crop the image to match the validation video dimensions
                    image = F.resize(image, size=min(width, height))
                    image = F.center_crop(image, output_size=(width, height))
                pipeline_inputs["image"] = image

            # Load and add reference video, if provided
            if self._config.validation.reference_videos is not None:
                video_path = self._config.validation.reference_videos[j]
                ref_video, _ = read_video(video_path)[:frames]
                pipeline_inputs["reference_video"] = ref_video

            with torch.amp.autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                result = pipeline(**pipeline_inputs)
                videos = result.frames

            for video in videos:
                video_path = output_dir / f"step_{self._global_step:06d}_{i}.mp4"
                export_to_video(video, str(video_path), fps=24)
                video_paths.append(video_path)
                i += 1
            progress.update(task, advance=1)

        progress.remove_task(task)

        # Move unused components back to CPU.
        self._vae.to("cpu")
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to("cpu")

        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        logger.info(f"🎥 Validation samples for step {self._global_step} saved in {rel_outputs_path}")
        return video_paths
