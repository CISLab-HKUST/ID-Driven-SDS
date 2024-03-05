import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torchvision.utils import save_image
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from torch.cuda.amp import custom_bwd, custom_fwd
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDIMScheduler, StableDiffusionControlNetPipeline
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from .sd_step import *
from controlnet_aux import ControlNetModel
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
    
def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator


class StableDiffusionIPAdapterGuidance:
    def __init__(self, device, config, use_controlnet=False):
        self.device = device
        self.config = config
        self.image_encoder_path = config.image_encoder_path
        self.vae_model_path = config.vae_model_path

        self.ip_ckpt = config.ip_ckpt
        self.torch_dtype = torch.float16 if config.fp16 else torch.float32
        self.insightface_path = config.insightface_path
        
        # load StableDiffusionPipleline
        self.base_model_path = config.base_model_path

        vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(dtype=self.torch_dtype)

        pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model_path,
            vae=vae,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        
        self.seed = config.seed
        self.generator = get_generator(self.seed, self.device)

        if config.vram_O:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.enable_attention_slicing(1)
            self.pipe.enable_model_cpu_offload()

        self.ip_adapter = IPAdapterFaceIDPlus(
            self.pipe,
            self.image_encoder_path,
            self.ip_ckpt,
            self.device,
            torch_dtype=self.torch_dtype
        )

        self.unet = self.ip_adapter.pipe.unet

        # TODO Skeleton ControlNet
        self.use_controlnet = use_controlnet
        self.controlnet_path = config.controlnet_path
        if use_controlnet:
            self.controlnet_skeleton = ControlNetModel.from_pretrained(
                self.controlnet_path,
                torch_dtype=self.torch_dtype
            ).to(self.device)
        else:
            self.controlnet_skeleton = None

        # timestep
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
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

        # ID Diven Related Images
        self.face_image_path = config.face_image_path
        self.faceid_embeds, self.face_image = self.prepare_face_images()

        # ID Scale
        self.scale = config.scale
        self.s_scale = config.s_scale
        self.shortcut = config.shortcut

        self.guidacen_scale = config.guidance_scale

        # embeddings related
        self.image_prompt_embeds, self.uncond_image_prompt_embeds = self.ip_adapter.get_image_embeds(self.faceid_embeds, self.face_image, self.s_scale, self.shortcut)

        print(f'[INFO] loaded guidance (stable diffusion + ip adapter)!')

    @torch.no_grad()
    def get_full_embeds(self, batch_size, full_embeddings):
        bs_embed, seq_len, _ = self.image_prompt_embeds.shape
        image_prompt_embeds = self.image_prompt_embeds.repeat(1, batch_size, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * batch_size, seq_len, -1)

        uncond_image_prompt_embeds = self.uncond_image_prompt_embeds.repeat(1, batch_size, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * batch_size, seq_len, -1)

        empty_prompt_embeds_, uncond_prompt_embeds_, cond_prompt_embeds_  = full_embeddings.chunk(3)
        empty_prompt_embeds = torch.cat([empty_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        uncond_prompt_embeds = torch.cat([uncond_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        cond_prompt_embeds = torch.cat([cond_prompt_embeds_, image_prompt_embeds], dim=1)
        
        return empty_prompt_embeds, uncond_prompt_embeds, cond_prompt_embeds

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def prepare_face_images(self):
        # prepare face images and embeddings
        app = FaceAnalysis(name="buffalo_l", root=self.insightface_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        image = cv2.imread(self.face_image_path)
        faces = app.get(image)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        # (224, 224, 3), numpy.ndarray
        face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224) # you can also segment the face

        return faceid_embeds, face_image
    

    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)
    
    def adjust_max_step(self, iteration):
        max_step_0 = self.max_step_0
        max_step_1 = self.max_step_1

        now_max_step = max_step_1 - int(iteration / self.total_iteration * (max_step_1 - max_step_0))

        return now_max_step
    
    @torch.no_grad()
    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence

    def train_step(
        self,
        full_embeddings,
        pred_rgb,
        pred_depth=None,
        pred_skeleton=None,
        iteration=None,
        resolution=(512,512),
        save_folder=None,
        vis_interval=20,
    ):
        batch_size = pred_rgb.shape[0]

        if self.anneal_timesteps:
            ind_t = int(self.max_step - (self.max_step - self.min_step) * math.sqrt(iteration / self.total_iteration))
        else:  
            self.max_step = self.adjust_max_step(iteration)
            ind_t = torch.randint(self.min_step, self.max_step, (pred_rgb.shape[0],), dtype=torch.long, device=self.device)[0]
        
        t = self.timesteps[ind_t]
        t = torch.tensor([t], dtype=torch.long, device=self.device)

        empty_embeds, uncond_embeds, cond_embeds = self.get_full_embeds(batch_size=batch_size, full_embeddings=full_embeddings)
        text_embeds = torch.cat([empty_embeds, uncond_embeds, cond_embeds], dim=0)

        noise = torch.randn(
            (pred_rgb.shape[0], 4, resolution[0] // 8, resolution[1] // 8),
            dtype=pred_rgb.dtype, device=pred_rgb.device, generator=self.generator
        )


        # TODO, add skeleton controlnet
        with torch.no_grad():
            latents_noisy = self.scheduler.add_noise(pred_rgb, noise, t)
            latent_model_input = latents_noisy[None, :, ...].repeat(3, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            if self.use_controlnet:
                pred_skeleton_input = pred_skeleton[None, :, ...].repeat(3, 1, 1, 1, 1).reshape(-1, 3, resolution[0] // 8, resolution[1] // 8, )
                down_block_res_samples, mid_block_res_sample = self.controlnet_skeleton(
                    latent_model_input,
                    tt,
                    encoder_hidden_states=text_embeds,
                    controlnet_cond=pred_skeleton_input,
                    return_dict=False,
                )
                unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeds,
                                    down_block_additional_residuals=down_block_res_samples,
                                    mid_block_additional_residual=mid_block_res_sample).sample
            
            else:
                unet_output = self.unet(
                    latent_model_input.to(self.torch_dtype), 
                    tt.to(self.torch_dtype), 
                    encoder_hidden_states=text_embeds.to(self.torch_dtype)
                ).sample

            noise_pred_empty, noise_pred_uncond, noise_pred_cond = unet_output.chunk(3)

            delta_c = self.guidacen_scale * (noise_pred_cond - noise_pred_empty)
            mask = (t < 200).int().view(pred_rgb.shape[0], 1, 1, 1)
            delta_d = mask * noise_pred_empty + (1 - mask) * (noise_pred_empty - noise_pred_uncond)
        
        noise_pred = delta_c + delta_d
        w = (1 - self.alphas[t.item()]).view(-1, 1, 1, 1)

        grad = w * (noise_pred)
        grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(pred_rgb, (pred_rgb - grad).detach(), reduction="mean") / pred_rgb.shape[0]

        if iteration % vis_interval == 0:
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration, t.item()))
            with torch.no_grad():
                labels = self.decode_latents((pred_rgb - grad).type(self.torch_dtype))
                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)
                pred_rgb = self.decode_latents(pred_rgb.detach())
                viz_images = torch.cat([pred_rgb, norm_grad,
                                        labels],dim=0) 
                save_image(viz_images, save_path_iter)

        return loss
