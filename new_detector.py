import cv2
import numpy as np
from torch import torch
from torchvision import transforms as T
from face_utils import norm_crop, FaceDetector
from model_def.new_xception import xception


class ImageLoader:
    def __init__(self, face_detector, transform=None):
        self.face_detector = face_detector
        self.transform = transform
        model = xception(parameters=True)
        model = model.cpu()
        model.eval()
        self.model = model

    def predictOneFace(self, img, landmarks):
        model = self.model

        img = norm_crop(img, landmarks, image_size=299)
        if self.transform:
            img = self.transform(img)

        batch_buf = []
        batch_buf.append(img)
        batch = torch.stack(batch_buf)
        batch = batch.cpu()
        output = model(batch, batch)
        output = torch.sigmoid(output)[:, 1]

        return output

    def predict(self, img):
        boxes, landms = self.face_detector.detect(img)
        if boxes.shape[0] == 0:
            return boxes, []
        scores = []
        for landmark in landms:
            # Tensor.detach() 阻断反向传播 ndarry.astype(np.int)  取整
            landmarks = landmark.reshape(5, 2).astype(np.int32)
            scores.append(self.predictOneFace(img, landmarks))

        return boxes, scores

    def predict_raw(self, img_raw):
        img = cv2.imdecode(np.frombuffer(img_raw, np.uint8), cv2.IMREAD_COLOR)
        return self.predict(img)


# load model
# 只要设置了torch.set_grad_enabled(False)那么接下来所有的tensor运算产生的新的节点都是不可求导的，
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
face_detector = FaceDetector()
face_detector.load_checkpoint("./input/dfdc-pretrained-2/RetinaFace-Resnet50-fixed.pth")
loader = ImageLoader(face_detector, T.ToTensor())
print("loader ok")
