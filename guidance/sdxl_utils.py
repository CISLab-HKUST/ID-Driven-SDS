import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from diffusers import DDIMScheduler
from torchvision.utils import save_image
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from torch.cuda.amp import custom_bwd, custom_fwd

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator

class StableDiffusionXLIPAdapterGuidance:
    def __init__(self, device, config, use_controlnet) -> None:
        self.device = device
        self.config = config
        self.base_model_path = config.base_model_path
        self.image_encoder_path = config.image_encoder_path
        self.ip_ckpt = config.ip_ckpt
        self.torch_dtype = torch.float16 if config.fp16 else torch.float32

        # get face region and face embedding
        # TODO