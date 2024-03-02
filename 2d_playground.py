import torch
import os
from guidance.sd_utils import StableDiffusionIPAdapterGuidance
from torchvision import transforms
from tqdm import tqdm
from torch.optim import Adam
from utils.images_to_video import imgs_to_video
from PIL import Image
from configs.guidance_config import GuidanceConfig

if __name__ == "__main__":
    device = torch.device("cuda:0")
    save_folder = "./exprs/test1"
    total_iteration = 1000
    batch_size = 1
    skeleton_image_path = "" # your skeleton image
    config = GuidanceConfig
    config.total_iteration = total_iteration

    prompt = "a woman wearing blue dress, full body, photorealistic, 8K, HDR."
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    empty_prompt = ""

    guidance = StableDiffusionIPAdapterGuidance(
        device=device,
        config=config,
    )

    empty_embeds = guidance.get_text_embeds(empty_prompt)
    uncond_embeds = guidance.get_text_embeds(negative_prompt)
    cond_embeds = guidance.get_text_embeds(prompt)

    empty_embeds = empty_embeds.repeat(batch_size, 1, 1)
    uncond_embeds = uncond_embeds.repeat(batch_size, 1, 1)
    cond_embeds = cond_embeds.repeat(batch_size, 1, 1)

    text_embeds = torch.cat([empty_embeds, uncond_embeds, cond_embeds], dim=0)

    skeleton_image = Image.open(skeleton_image_path).resize((512, 512), Image.Resampling.BILINEAR).convert("RGB")
    transf = transforms.ToTensor()
    skeleton_img_tensor = transf(skeleton_image).unsqueeze(0)

    img_tensor = torch.full((1, 3, 512, 512), 255).repeat(batch_size, 1, 1, 1)
    img_latents, _ = guidance.encode_imgs(img_tensor.to(torch.float32).to(device))
    img_latents.requires_grad = True

    image_save_folder = os.path.join(save_folder, "images")
    os.makedirs(image_save_folder, exist_ok=True)
    video_save_folder = os.path.join(save_folder, "video")
    os.makedirs(video_save_folder, exist_ok=True)

    optimizer = Adam([
        {'params': img_latents, 'lr': 1e-2}
    ])

    for iteration in tqdm(range(total_iteration)):
        loss = guidance.train_step(
            full_embeddings=text_embeds,
            pred_rgb=img_latents,
            pred_skeleton=skeleton_img_tensor,
            iteration=iteration,
            save_folder=image_save_folder,
            vis_interval=20
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    imgs_to_video(image_save_folder, os.path.join(video_save_folder, "videos.mp4"))