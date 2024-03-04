import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL
from torchvision.utils import save_image
from PIL import Image
from controlnet_aux import OpenposeDetector
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from diffusers import StableDiffusionXLControlNetPipeline

app = FaceAnalysis(name="buffalo_l", root="D:/programming/huggingface/cache/annotator/insightface", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

image = cv2.imread("images/woman.png")
faces = app.get(image)

faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224) # you can also segment the face


base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "D:/programming/huggingface/cache/vae/sd-vae-ft-mse"
image_encoder_path = "D:/programming/huggingface/cache/vae/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "D:/programming/huggingface/cache/adapter/ip-adapter/sd15/ip-adapter-faceid-plusv2_sd15.bin"
device = "cuda"

checkpoint = "lllyasviel/control_v11p_sd15_openpose"

image = load_image(
    "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/input.png"
)


processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

control_image = processor(image, hand_and_face=True)
control_image.save("./images/control.png")

controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    controlnet=controlnet,
    feature_extractor=None,
    safety_checker=None
)

# load ip-adapter
ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)

# generate image
prompt = "photo of a woman in red dress in a garden"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

image = ip_model.generate(
     prompt=prompt, negative_prompt=negative_prompt, face_image=face_image, faceid_embeds=faceid_embeds, shortcut=True, s_scale=1.0,
     image=control_image, num_samples=3, width=512, height=768, num_inference_steps=30, seed=2023
)[0]
image.save('test.png')
