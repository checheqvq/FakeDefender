import cv2
import numpy as np
from PIL import Image
from torch import torch
import torch.nn.functional as F
from torchvision import transforms as T
from face_utils import norm_crop, FaceDetector
from model_def.new_xception import xception


class DFDCImageLoader:
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


        # 转换成张量
        if self.transform:
            img = self.transform(img)

        batch_buf = []
        batch_buf.append(img)
        # torch.stack()是将原来的几个tensor按照一定方式进行堆叠，然后在按照堆叠后的维度进行切分。
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
# 这个相当于一个全局的环境，即使是多个循环或者是在函数内设置的调用，
# 只要torch.set_grad_enabled(False)出现，则不管是在下一个循环里还是在主函数中，都不在求导，
# 除非单独设置一个孤立节点，并把他的requires_grad设置成true。
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
torch.set_grad_enabled(False)
# 让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
torch.backends.cudnn.benchmark = True
# 加载检查点
face_detector = FaceDetector()
# RetinaFace-Resnet50-fixed.pth-->Pretrained RetinaFace模型
face_detector.load_checkpoint("./input/dfdc-pretrained-2/RetinaFace-Resnet50-fixed.pth")
# T.ToTensor() 把图片转换成张量的形式  print(T.ToTensor()) --> ToTensor()
loader = DFDCImageLoader(face_detector, T.ToTensor())
print("loader ok")
