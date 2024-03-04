import torch
import numpy as np
from PIL import Image

def image_to_tensor(image):
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array) / 255.
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    return image_tensor