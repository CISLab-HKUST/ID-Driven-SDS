import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torchvision.utils import save_image
from diffusers import DDIMScheduler, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline
from torchvision.utils import save_image
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from torch.cuda.amp import custom_bwd, custom_fwd
from controlnet_aux import ControlNetModel

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from ip_adapter.ip_adapter import IPAdapterPlusXL


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
    def __init__(self, device, config, use_controlnet=False) -> None:
        self.device = device
        self.config = config
        self.base_model_path = config.base_model_path
        self.image_encoder_path = config.image_encoder_path
        self.controlnet_model_path = config.controlnet_model_path
        self.ip_ckpt = config.ip_ckpt
        self.torch_dtype = torch.float16 if config.fp16 else torch.float32

        self.seed = config.seed
        self.generator = get_generator(self.seed, self.device)

        if use_controlnet:
            controlnet_skeleton = ControlNetModel.from_pretrained(
                self.controlnet_model_path,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=self.torch_dtype,
                controlnet=controlnet_skeleton,
            ).to(self.device)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=self.torch_dtype
            ).to(self.device)

        if config.vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        self.ip_adapter_xl = IPAdapterPlusXL(
            sd_pipe=pipe,
            image_encoder_path=self.image_encoder_path,
            ip_ckpt=self.ip_ckpt,
            device=self.device,
            num_tokens=16,
        )

        self.pipe = self.ip_adapter_xl.pipe

        self.unet = self.pipe.unet

        if use_controlnet:
            self.controlnet = self.pipe.controlnet
        else:
            self.controlnet = None


        self.scheduler = self.pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))

        min_range = config.min_range
        # [0.5, 0.98]
        max_range = config.max_range
        self.min_step = int(self.num_train_timesteps * min_range)
        self.max_step = int(self.num_train_timesteps * max_range[1])
        self.max_step_0 = int(self.num_train_timesteps * max_range[0])
        self.max_step_1 = int(self.num_train_timesteps * max_range[1])

        self.anneal_timesteps = config.anneal_timesteps
        self.total_iteration = config.total_iteration

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        # ID Face Image
        # TODO ID Given Image to Image
        self.face_image_path = config.face_image_path

        # get face region and face embedding
        # TODO

    '''
    input for SDXL pipeline have 4 attribute
    1: Image Embeddings
    2.: Text Embeddings
        1 + 2 will be the final prompt embeddings
    3: pooled text embeddings - add_text_embeddings
    4. add_time_ids
        3 + 4 will be the added_cond_kwargs
    '''

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt=""):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    
    @torch.no_grad()
    def get_null_embeds(self, cond_embeds, pooled_prompt_embeds):
        torch_dtype = cond_embeds.dtype
        device = cond_embeds.device
        
        null_embeds = torch.zeros_like(cond_embeds).to(torch_dtype).to(device)
        pooled_null_embeds = torch.zeros_like(pooled_prompt_embeds).to(torch_dtype).to(device)

        return null_embeds, pooled_null_embeds
    
    @torch.no_grad()
    def get_image_embeds(self, image_tensor):
        '''
        should get the id face image embeddings for ID consistency (to be analysed)
        should get the render image embeddings for reduce variance
        '''
        # NOTE for the rendered image, not include id face image 
        image_prompt_embeds, uncond_image_prompt_embeds = self.ip_adapter_xl.get_image_embeds(image_tensor, do_rescale=False)

        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.no_grad()
    def get_face_image_embeds(self, face_image, batch_size):
        image_prompt_embeds, uncond_image_prompt_embeds = self.ip_adapter_xl.get_image_embeds(face_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape

        image_prompt_embeds = self.image_prompt_embeds.repeat(1, batch_size, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * batch_size, seq_len, -1)

        uncond_image_prompt_embeds = self.uncond_image_prompt_embeds.repeat(1, batch_size, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * batch_size, seq_len, -1)

        return image_prompt_embeds, uncond_image_prompt_embeds


    @torch.no_grad()
    def get_full_embeds(self, full_prompt_embeddings, image_tensor):
        # TODO Consider ID Face Image Embeddings
        empty_prompt_embeds_, uncond_prompt_embeds_, cond_prompt_embeds_  = full_prompt_embeddings.chunk(3)
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(image_tensor)

        empty_prompt_embeds = torch.cat([empty_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        uncond_prompt_embeds = torch.cat([uncond_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        cond_prompt_embeds = torch.cat([cond_prompt_embeds_, image_prompt_embeds], dim=1)

        return empty_prompt_embeds, uncond_prompt_embeds, cond_prompt_embeds


    @torch.no_grad()
    def get_add_time_ids(self, img_latents):
        _, _, H, W = img_latents.shape
        
        text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

        add_time_ids = self.pipe._get_add_time_ids(
            original_size=(128, 128),
            crops_coords_top_left=(0, 0),
            target_size=(128, 128),
            dtype=self.torch_dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        return add_time_ids