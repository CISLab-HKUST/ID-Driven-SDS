import torch
import torch.nn.functional as F
import numpy as np
import os
import shutil
from torchvision.utils import save_image
from face_process.face_processor import FaceProcessor
from PIL import Image
from torch.optim import Adam
from tqdm import tqdm
from utils.images_to_video import imgs_to_video_simple
from torchviz import make_dot
from utils.common import *

def condition_cosine_loss(input_a, input_b, condition=0.5):
    pass

if __name__ == "__main__":

    device = torch.device("cuda")
    torch_dtype = torch.float32
    detect_model_path = "weights/Retinaface_mobilenet0.25.pth"
    recognize_model_path = "weights/arcface_mobilenet_v1.pth"
    label_image_path = "label.png"
    train_image_path = "assets/test_blue.png"
    # train_image_path = 'image_after_id_loss.png'
    output_dir = "face_similar_loss"

    os.makedirs(output_dir, exist_ok=True)

    total_iteration = 500

    face_processor = FaceProcessor(
        device=device,
        torch_dtype=torch_dtype,
        detect_model_path=detect_model_path,
        recognize_model_path=recognize_model_path,
    )

    label_image = Image.open(label_image_path).convert("RGB")
    train_image = Image.open(train_image_path).convert("RGB")
    # print(train_face_feature.shape)

    # cos_similarity = F.cosine_similarity(label_face_feature, train_face_feature)
    # print(f"cos similarity: {cos_similarity}")

    # train image should be set to learnable tensor
    # train_image_tensor = image_to_tensor(train_image).to(torch_dtype).to(device)
    
    train_image_tensor = torch.zeros((1, 3, 112, 112), dtype=torch_dtype, device=device)
    train_image_tensor_copy = train_image_tensor.clone()
    train_image_tensor.requires_grad = True

    optimizer = Adam([
        {'params': train_image_tensor, 'lr': 1e-2}
    ])

    label_image_tensor = image_to_tensor(label_image)
    label_face_box = [[0,0,label_image_tensor.shape[2], label_image_tensor.shape[3]]]
    label_face_feature = face_processor.recognize(label_image_tensor, label_face_box[0], pre_defined=False)
    # _, train_face_feature = face_processor.process(train_image, pre_defined=True)
    # train_face_box, train_face_feature = face_processor.process(train_image_tensor, pre_defined=False)
    train_face_box = [[0, 0, 112, 112]]
    # cos_similarity = F.cosine_similarity(train_face_feature, label_face_feature)
    # print(cos_similarity)

    # exit(0)
    print(label_face_feature)
    pbar = tqdm(range(total_iteration))
    for iteration in pbar:

        # _, train_face_feature = face_processor.process(train_image_tensor, pre_defined=False)
        train_face_feature = face_processor.recognize(train_image_tensor, box=train_face_box[0], pre_defined=False)

        loss = F.mse_loss(train_face_feature, label_face_feature.detach(), reduction="sum")

        cos_similarity = F.cosine_similarity(train_face_feature, label_face_feature)

        tensor_differences = round((train_image_tensor - train_image_tensor_copy).sum().item(), 3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"marginal_cosine_loss:{round(loss.item(), 3)}, face similarity: {round(cos_similarity.item(), 3)}, tensor_differences: {tensor_differences}")

        if iteration % 10 == 0:
            save_image(train_image_tensor, os.path.join(output_dir, f"{iteration}.png"))

    imgs_to_video_simple(output_dir, "train_face_loss.mp4")
    shutil.rmtree(output_dir)
    torch.save(train_image_tensor, "tensor.pth")
    save_image(train_image_tensor, "image_after_id_loss.png")
