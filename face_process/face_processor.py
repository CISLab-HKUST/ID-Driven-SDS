import torch
import torch.nn.functional as F
from .utils import process_box
from face_process.retinaface.face_detector import FaceDetector
from face_process.arcface.face_recognizer import FaceRecognizer

class FaceProcessor(object):
    def __init__(
        self,
        device,
        torch_dtype,
        detect_model_path,
        recognize_model_path,
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.detect_model_path = detect_model_path
        self.recognize_model_path = recognize_model_path

        self.generate()

    @torch.no_grad()
    def generate(self):
        print("[Face Processor] Loading Weights...")

        self.face_detector = FaceDetector(
            device=self.device,
            torch_dtype=self.torch_dtype,
            model_path=self.detect_model_path,
        )

        self.face_recognizer = FaceRecognizer(
            device=self.device,
            torch_dtype=self.torch_dtype,
            model_path=self.recognize_model_path,
        )

    # test code, only for batch_size=1
    def recognize(self, image, box, pre_defined=False):
        batch = 1 if pre_defined else image.shape[0]
        input_shape = [112, 112, 3]
        face_box = process_box(box)
        if pre_defined == True:
            face_region = image.crop(face_box)
            face_feature = self.face_recognizer.detect_image(face_region, pre_defined=pre_defined)

        else:
            # print(image.shape)
            # print(face_box)
            face_region = image[0, :, face_box[0]:face_box[2], face_box[1]:face_box[3]][None]
            face_region = F.interpolate(face_region, (input_shape[0], input_shape[1]), mode='bilinear', align_corners=None)
            face_feature = self.face_recognizer.detect_image(face_region, pre_defined=pre_defined)

        return face_feature

    def process(self, image, pre_defined=False):
        # pred defined, arcface face features
        input_shape = [112, 112, 3]

        face_boxes = []
        face_features = []

        # one image just get one box
        batch = 1 if pre_defined else image.shape[0]

        for i in range(batch):
            # face_box prediction
            with torch.no_grad():
                if pre_defined == True:
                    region_pred = self.face_detector.detect_image(image, pre_defined=pre_defined)
                else:
                    region_pred = self.face_detector.detect_image(image[i][None], pre_defined=pre_defined)

                face_box = (region_pred[0][0], region_pred[0][1], region_pred[0][2], region_pred[0][3])
                face_box = process_box(face_box)
                face_boxes.append(face_box)

            if pre_defined == True:
                face_region = image.crop(face_box)
                face_feature = self.face_recognizer.detect_image(face_region, pre_defined=pre_defined)

            else:
                # print(image.shape)
                # print(face_box)
                face_region = image[i, :, face_box[0]:face_box[2], face_box[1]:face_box[3]][None]
                face_region = F.interpolate(face_region, (input_shape[0], input_shape[1]), mode='nearest', align_corners=None)
                face_feature = self.face_recognizer.detect_image(face_region, pre_defined=pre_defined)

            face_features.append(face_feature)

        # face_box = torch.cat(face_boxes, dim=0, dtype=torch.int32)
        face_features = torch.cat(face_features, dim=0).to(self.torch_dtype).to(self.device)
        # print(f"face_features shape: {face_features.shape}")

        return face_boxes, face_features