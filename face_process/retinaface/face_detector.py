import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from .nets.retinaface import RetinaFace
from .utils.anchors import Anchors
from .utils.config import cfg_mnet, cfg_re50
from .utils.utils import letterbox_image, preprocess_input
from .utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)

class FaceDetector(object):
    def __init__(
        self, 
        device,
        torch_dtype,
        model_path, 
        backbone="mobilenet", 
        confidence=0.5, 
        nms_iou=0.45,
        input_shape=[1280, 1280, 3],
    ) -> None:
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_path = model_path
        self.backbone = backbone
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.input_shape = input_shape

        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        self.generate()

    def generate(self):
        print('[RetinaFace] Loading weights into state dict...')
        self.net = RetinaFace(cfg=self.cfg, mode='eval').eval()
        device = self.device

        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        
        self.net = nn.DataParallel(self.net).to(self.device)
        for p in self.net.parameters():
            p.requires_grad = False

    def preprocess_pre_defined(self, image):
        self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors().to(self.device)
        image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).to(self.torch_dtype).to(self.device)

        return image

    def detect_image(self, image, pre_defined=False):
        '''
        pre_defined: ID Image (PIL.Image) or Rendered Image (tensor: B C H W)
        '''
        if pre_defined:
            image = np.array(image, dtype=np.float32 if self.torch_dtype is torch.float32 else np.float16)
            H, W, C = image.shape
            image = self.preprocess_pre_defined(image)

        else:
            B, C, H, W = image.shape
            # image_tensor = (torch.from_numpy(np.array(image)) - 127.5)
            image = (image * 255. - 127.5).to(self.torch_dtype).to(self.device)
            self.anchors = Anchors(self.cfg, image_size=(H, W)).get_anchors().to(self.device)

        scale = [
            W, H, W, H
        ]
        scale_for_landmarks = [
            W, H, W, H,
            W, H, W, H,
            W, H
        ]
        loc, conf, landms = self.net(image)
        boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
        conf    = conf.data.squeeze(0)[:, 1:2]
        landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])

        boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
        boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

        # TODO Better Solution? Just pick the most likely one
        # boxes_conf_landms = boxes_conf_landms[0][None]

        if len(boxes_conf_landms) <= 0:
            return None
        
        if pre_defined:
            boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                    np.array([self.input_shape[0], self.input_shape[1]]), np.array([H, W]))
        
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        return boxes_conf_landms


