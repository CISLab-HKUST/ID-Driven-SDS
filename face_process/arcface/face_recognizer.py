import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


from .nets.arcface import Arcface as arcface
from .utils.utils import preprocess_input, resize_image, show_config

class FaceRecognizer(object):
    def __init__(
        self,
        device,
        torch_dtype,
        model_path,
        backbone="mobilenetv1",
        input_shape=[112, 112, 3],
        letterbox_image=True
    ) -> None:
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_path = model_path
        self.backbone = backbone
        self.input_shape=input_shape
        self.letterbox_image = letterbox_image

        self.generate()

    def generate(self):
        print('[ArcFace] Loading weights into state dict...')
        self.net = arcface(backbone=self.backbone, mode="predict").eval()
        device = self.device
        
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net = self.net.eval()

        self.net = nn.DataParallel(self.net).to(self.device)

        for p in self.net.parameters():
            p.requires_grad = False

    def preprocess_pre_defined(self, image):
        image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=self.letterbox_image)
        
        photo = torch.from_numpy(
            np.expand_dims(
                np.transpose(
                    preprocess_input(
                        np.array(image, dtype=np.float32 if self.torch_dtype is torch.float32 else np.float16)
                        ), (2, 0, 1)
                ), 
            0)
        ).to(self.device)

        return photo

    def detect_image(self, image, pre_defined=False):

        if pre_defined:
            image = self.preprocess_pre_defined(image)
        else:
            image = (image - 0.5) / 0.5
            image = image.to(self.torch_dtype).to(self.device)

        output = self.net(image)

        return output