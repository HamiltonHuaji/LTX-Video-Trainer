import math
import copy
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from einops import *
from typing import *
from jaxtyping import Float, Shaped, Int, Integer, Bool
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder
from pydantic import BaseModel, computed_field
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Group, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import torchvision.transforms.functional

from ltxv_trainer import logger
from ltxv_trainer.trainer import LtxvTrainer
from ltxv_trainer.config import *
from ltxv_trainer.training_strategies import get_training_strategy, TrainingStrategy, TrainingBatch, DEFAULT_FPS
from ltxv_trainer.timestep_samplers import TimestepSampler
from ltxv_trainer.model_loader import LtxvModelComponents, LtxvModelVersion, ModelSource, try_parse_version, _is_safetensors_url, _is_huggingface_repo, load_ltxv_components, load_scheduler, load_text_encoder, load_tokenizer, load_vae, load_transformer
from ltxv_trainer.quantization import quantize_model
from ltxv_trainer.ltxv_utils import encode_video, decode_video, get_rope_scale_factors, prepare_video_coordinates, VideoEncoding
from ltxv_trainer.utils import open_image_as_srgb
from ltxv_trainer.video_utils import read_video
from diffusers import LTXVideoTransformer3DModel as _LTXVideoTransformer3DModel
from diffusers.models.autoencoders import AutoencoderKLLTXVideo

from dinov3.models.vision_transformer import DinoVisionTransformer
from gshub.utils import not_none_or_dotenv, describe

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

def dinov3_transform(pixels: Float[torch.Tensor, 't c h w'], patch_size: int = 16) -> Float[torch.Tensor, 't c h w']:
    t, c, h, w = pixels.shape
    if (h % patch_size != 0) or (w % patch_size != 0):
        resize_hw = round(h / patch_size) * patch_size, round(w / patch_size) * patch_size
        pixels = F.interpolate(pixels, size=resize_hw, mode='bicubic', align_corners=False)
    return torchvision.transforms.functional.normalize(pixels, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def dinov3_feature(dinov3: DinoVisionTransformer, pixels: Float[torch.Tensor, '... c h w'], patch_size: int = 16) -> Float[torch.Tensor, '... c h w']:
    # with torch.autocast('cuda', torch.bfloat16):
    image_encoder_hidden_states, = dinov3.get_intermediate_layers(dinov3_transform(rearrange(pixels, '... c h w -> (...) c h w'), patch_size=patch_size), n=1, reshape=True)
    image_encoder_hidden_states = cast(torch.Tensor, image_encoder_hidden_states)
    image_encoder_hidden_states = image_encoder_hidden_states.view(*pixels.shape[:-3], *image_encoder_hidden_states.shape[-3:]) # ... c h w
    return image_encoder_hidden_states

class CustomValidationConfig(ValidationConfig):
    condition_videos: list[str] | None = Field(
        default=None,
        description="List of reference video paths to use for validation. "
        "One video path must be provided for each validation prompt",
    )

class CustomTrainerConfig(LtxvTrainerConfig):
    validation: CustomValidationConfig = Field(default_factory=CustomValidationConfig)
    dinov3_model_name: str = Field(default='dinov3_vitl16')
    dinov3_repository: str | None = Field(default=None)
    dinov3_checkpoint: str | None = Field(default=None)
    dinov3_source: str = Field(default='local')

def get_dinov3(config: CustomTrainerConfig):
    dinov3_source = not_none_or_dotenv(config.dinov3_source, 'dinov3_source')
    dinov3_repository = not_none_or_dotenv(config.dinov3_repository, 'dinov3_repository')
    dinov3_checkpoint = not_none_or_dotenv(config.dinov3_checkpoint, 'dinov3_vitl16_pretrain_lvd1689m')
    return cast(DinoVisionTransformer, torch.hub.load(dinov3_repository, MODEL_DINOV3_VITL, source=dinov3_source, weights=dinov3_checkpoint))

def encode_video_as_image(vae: AutoencoderKLLTXVideo, video: Float[torch.Tensor, '... c h w']) -> Float[torch.Tensor, '... h w c']:
    video_shape = video.shape
    video = rearrange(video, '... c h w -> (...) 1 c h w')
    latents_dict = encode_video(vae, video.to(vae.device))
    latents = latents_dict['latents'] # (...) (1 h w) c
    num_channels = latents.size(-1)
    num_frames = latents_dict['num_frames'] # must be 1
    assert num_frames == 1

    height = latents_dict['height'] # compressed but not patchified height, equals to h // 32
    width = latents_dict['width'] # compressed but not patchified height, equals to w // 32
    latents = latents.view(*video_shape[:-3], height, width, num_channels)
    return latents

def encode_video_as_image_dict(vae: AutoencoderKLLTXVideo, video: Float[torch.Tensor, 'b f c h w']) -> VideoEncoding:
    latents = encode_video_as_image(vae, video) # b f h w c
    num_frames = latents.size(1)
    height = latents.size(2)
    width = latents.size(3)
    return {
        'latents': rearrange(latents, 'b f h w c -> b (f h w) c'),
        'num_frames': num_frames,
        'height': height,
        'width': width
    }

import decord
import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, resize, to_tensor
def ltxv_resize_and_crop(arr: torch.Tensor, image_size: tuple[int, int], reshape_mode='center') -> torch.Tensor:
    """Resize and crop tensor to target size."""
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(arr, size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])], interpolation=InterpolationMode.BICUBIC)
    else:
        arr = resize(arr, size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]], interpolation=InterpolationMode.BICUBIC)
    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)
    delta_h = h - image_size[0]
    delta_w = w - image_size[1]
    if reshape_mode == "random":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise ValueError(f"Unsupported reshape mode: {reshape_mode}")
    arr = crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr

def ltxv_preprocess_video(path: Path, num_frames=41, resolution=(640, 960)) -> tuple[torch.Tensor, float]:
    """Preprocess a video by loading, resizing, and applying transforms."""
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Lambda(lambda x: x.clamp_(0, 1)),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    video_reader = decord.VideoReader(uri=path.as_posix())
    video_num_frames = len(video_reader)
    fps = video_reader.get_avg_fps()

    frame_indices = list(range(video_num_frames))
    frames = video_reader.get_batch(frame_indices)
    if isinstance(frames, decord.ndarray.NDArray):
        frames = torch.from_numpy(frames.asnumpy())
    frames = frames[:num_frames].float() / 255.0
    frames = frames.permute(0, 3, 1, 2).contiguous()

    frames_resized = ltxv_resize_and_crop(frames, resolution)
    frames = torch.stack([transforms(frame) for frame in frames_resized], dim=0)

    return frames, fps

class CustomDL3DV10KDatasetBatch(TypedDict):
    target_pixels: Float[torch.Tensor, 'b t c h w']
    fps: float | None
    prompt_embeds: Float[torch.Tensor, 'b max_length d']
    prompt_attention_mask: Bool[torch.Tensor, 'b max_length']

    # reference: 指负责视角控制、与输出对齐的视频

    reference_pixels: Float[torch.Tensor, 'b t c h w']
    reference_depths: Float[torch.Tensor, 'b t c h w']
    reference_extrinsics: Float[torch.Tensor, 'b t 4 4']
    reference_intrinsics: Float[torch.Tensor, 'b t 3 3']

    # condition: 指额外的其它视角的信息

    condition_pixels: Float[torch.Tensor, 'b r c h w']
    condition_extrinsics: Float[torch.Tensor, 'b r 4 4']
    condition_intrinsics: Float[torch.Tensor, 'b r 3 3']

class CustomDL3DV10KDataset(Dataset):
    def __init__(
        self, data_root: str | Path, num_frames: int, num_cond_frames: int, resolution: Tuple[int, int] = (640, 960), *,
        discrete_reference_indices: bool | float = False, discrete_condition_indices: bool | float = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        assert self.data_root.is_dir(), f"Data root {data_root} is not a directory."
        self.items = [d for d in self.data_root.iterdir() if (d / '.done').exists()]
        self.items = [d for d in self.items if (d / 'prompt.pt').is_file()]
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
        slice_start = random.randint(0, total_length - length)
        slice_stop = slice_start + self.num_frames
        return slice(slice_start, slice_stop)

    def random_indices(self, length: int, total_length: int) -> List[int]:
        return torch.randperm(total_length)[:length].tolist()

    def __getitem__(self, index: int):
        item_dir = self.items[index % len(self)]
        # camera_params = torch.load(item_dir / 'camera_params.pth')
        # extrinsics: Float[torch.Tensor, 'n 4 4'] = camera_params['extrinsics'] # n 4 4, c2w
        # intrinsics: Float[torch.Tensor, 'n 3 3'] = camera_params['intrinsics'] # n 3 3
        # total_frames = intrinsics.size(0)

        ground_truth_decoder = VideoDecoder(item_dir / 'ground_truth.mp4')
        ref_pixels_decoder = VideoDecoder(item_dir / 'render_pixels.mp4')
        # ref_depths_decoder = VideoDecoder(item_dir / 'render_depths.mp4')
        total_frames = min(ground_truth_decoder.metadata.num_frames, ref_pixels_decoder.metadata.num_frames)

        if total_frames < self.num_frames:
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
            'fps': ground_truth_decoder.metadata.average_fps or DEFAULT_FPS,
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


T = TypeVar('T', bound=_LTXVideoTransformer3DModel)
def load_ltxv_transformer_custom(
    transformer_cls: type[T],
    source: ModelSource,
    low_cpu_mem_usage: bool = False, # if True, model will be created in meta device, which will cause problems when custom module of transformer exists
    torch_dtype = torch.bfloat16,
    **kwargs
) -> T:
    if isinstance(source, str):  # noqa: SIM102
        if version := try_parse_version(source):
            source = version

    if isinstance(source, LtxvModelVersion):
        assert source not in (LtxvModelVersion.LTXV_13B_097_DEV, LtxvModelVersion.LTXV_13B_097_DISTILLED), f"Untested: {source}"
        source_file = source.safetensors_url
    elif isinstance(source, (str, Path)):
        if _is_safetensors_url(source):
            source_file = str(source)
        elif _is_huggingface_repo(source):
            return transformer_cls.from_pretrained(source, subfolder="transformer", low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype, **kwargs)

        return transformer_cls.from_single_file(source_file, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype, **kwargs)

    raise ValueError(f"Invalid model source: {source}")

def load_ltxv_xfmr_2b_manually(transformer_cls: type[T], **kwargs):
    from diffusers.loaders.single_file_model import load_single_file_checkpoint
    from diffusers.loaders.single_file_utils import convert_ltx_transformer_checkpoint_to_diffusers, fetch_diffusers_config
    checkpoint = load_single_file_checkpoint("https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-dev-04-25.safetensors")
    config = fetch_diffusers_config(checkpoint)
    checkpoint = convert_ltx_transformer_checkpoint_to_diffusers(checkpoint)

    # from ltxv_trainer.custom.dinov3_proj_as_text.transformer_ltx import LTXVideoTransformer3DModel
    diffusers_model_config = cast(Dict[str, Any], transformer_cls.load_config(
        pretrained_model_name_or_path=config['pretrained_model_name_or_path'],
        subfolder='transformer'
    ))
    ltxv_xfmr = cast(T, transformer_cls.from_config(diffusers_model_config, **kwargs))
    missing, unexpected = ltxv_xfmr.load_state_dict(checkpoint, strict=False)
    if unexpected:
        print(f"Unexpected checkpoint keys: {unexpected}")

    return ltxv_xfmr
