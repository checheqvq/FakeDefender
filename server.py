import os
import sys
import time

import cv2
sys.path.append('/tmp/pycharm_project_deepfake')
sys.path.append('/tmp/pycharm_project_deepfake/external/EfficientNet-PyTorch')
sys.path.append('/tmp/pycharm_project_deepfake/external/Pytorch_Retinaface')
sys.path.append('/tmp/pycharm_project_deepfake/model_def')
sys.path.append('/home/yiboliu/.pycharm_helpers/pycharm_display')
sys.path.append('/home/yiboliu/.pycharm_helpers/pycharm_matplotlib_backend')
import numpy as np
from PIL import Image
from torch import torch
import torch.nn.functional as F
from torchvision import transforms as T
from face_utils import norm_crop, FaceDetector
from model_def import WSDAN, xception
from flask import Flask, jsonify, request
from rotation_test import rotation_dealer
import json
import base64

app = Flask(__name__)


class DFDCImageLoader:
    def __init__(self, face_detector, transform=None):
        self.face_detector = face_detector
        self.rotation_dealer = rotation_dealer()
        self.transform = transform
        # tensor view函数返回一个新tensor与源tensor共享内存，不改变数据，只改变形状
        # 重构张量维度 四维张量 张量 解释 https://www.jianshu.com/p/f34457c222c5
        # view(数量，行，列，深度) (2, 3, 4, 2) 两个3行4列深度为2的张量
        # 第一个方括号——深度上位于同一位置的数据——对应参数第4个
        # 第二个方括号——列数据——对应参数第3个
        # 第三个方括号——行数据——对应参数第2个，
        # 第四个方括号——数量——对应参数第1个
        self.zhq_nm_avg = torch.Tensor([.4479, .3744, .3473]).view(1, 3, 1, 1).cuda(0)
        self.zhq_nm_std = torch.Tensor([.2537, .2502, .2424]).view(1, 3, 1, 1).cuda(0)

        # Xception预训练权重文件
        # xception 引用自model_def文件夹
        model1 = xception(num_classes=2, pretrained=False)
        ckpt = torch.load("./input/dfdc-pretrained-2/xception-hg-2.pth")
        model1.load_state_dict(ckpt["state_dict"])
        # type(model1) --> model_def.xception.Xception
        model1 = model1.cuda(0)
        model1.eval()

        # WS-DAN w/ Xception的预训练权重文件
        model2 = WSDAN(num_classes=2, M=8, net="xception", pretrained=False).cuda(0)
        ckpt = torch.load("./input/dfdc-pretrained-2/ckpt_x.pth")
        model2.load_state_dict(ckpt["state_dict"])
        model2.eval()

        # WS-DAN w/ EfficientNet-b3的预训练权重文件
        model3 = WSDAN(num_classes=2, M=8, net="efficientnet", pretrained=False).cuda(0)
        ckpt = torch.load("./input/dfdc-pretrained-2/ckpt_e.pth")
        model3.load_state_dict(ckpt["state_dict"])
        model3.eval()

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def predictOneFace(self, img, landmarks):
        model1 = self.model1
        model2 = self.model2
        model3 = self.model3

        img = norm_crop(img, landmarks, image_size=320)
        # Image.fromarray()  numpy.ndarray转PIL.Image.Image
        aligned = Image.fromarray(img[:, :, ::-1])  # 将BGR转回RGB
        # aligned.show()

        # 转换成张量
        if self.transform:
            aligned = self.transform(aligned)

        batch_buf = []
        batch_buf.append(aligned)
        # torch.stack()是将原来的几个tensor按照一定方式进行堆叠，然后在按照堆叠后的维度进行切分。
        batch = torch.stack(batch_buf)
        batch = batch.cuda(0)

        # bilinear 双线性插值，
        i1 = F.interpolate(batch, size=299, mode="bilinear")
        i1.sub_(0.5).mul_(2.0)
        o1 = model1(i1).softmax(-1)[:, 1].cpu().detach().numpy()

        i2 = (batch - self.zhq_nm_avg) / self.zhq_nm_std
        o2, _, _ = model2(i2)
        o2 = o2.softmax(-1)[:, 1].cpu().detach().numpy()

        i3 = F.interpolate(i2, size=300, mode="bilinear")
        o3, _, _ = model3(i3)
        o3 = o3.softmax(-1)[:, 1].cpu().detach().numpy()

        out = 0.2 * o1 + 0.7 * o2 + 0.1 * o3
        return out[0]

    def predict(self, img):
        boxes, landms = self.face_detector.detect(img)
        if boxes.shape[0] == 0:
            return boxes, []
        boxes, landms = self.rotation_dealer.rotation_dealer(img, boxes, landms)
        scores = []
        for landmark in landms:
            # Tensor.detach() 阻断反向传播 ndarry.astype(np.int)  取整
            landmarks = landmark.reshape(5, 2).astype(np.int32)
            scores.append(self.predictOneFace(img, landmarks))

        return boxes, scores


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


@app.route('/', methods=['POST', 'GET'])
def hello():
    print("test")
    return "Welcome to Deepfake Server."


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    if request.method == 'POST':
        data = request.get_json()
        # data["image"] 是str类型
        data = json.loads(data,strict=False)
        img_encode = base64.b64decode(data["image"])
        # np.frombuffer() frombuffer将data以流的形式读入转化成ndarray对象
        # 第一参数为stream,第二参数为返回值的数据类型，第三参数指定从stream的第几位开始读入
        # np.uint8是用opencv处理图像时，其获得的矩阵类型是uint8
        # cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
        # type(img) --> numpy.ndarray
        img = cv2.imdecode(np.frombuffer(img_encode, np.uint8), cv2.IMREAD_COLOR)
        # faces 即是 boxes的取值
        faces, scores = loader.predict(img)


        response = {
            "faceNum": len(faces)
        }
        faceList = []
        if len(faces) != 0:
            for i in range(len(faces)):
                face = faces[i]
                faceList.append({
                    # item() 取出张量具体位置的元素元素值，并且返回的是该位置元素值的高精度值，保持原元素类型不变
                    "x1": int(face[0].item()),
                    "y1": int(face[1].item()),
                    "x2": int(face[2].item()),
                    "y2": int(face[3].item()),
                    "score": scores[i].item()
                })

        response["faces"] = faceList

        # 注释的代码是看打框后的图片
        # cv2.rectangle(img, (faceList[0]["x1"], faceList[0]["y1"]), (faceList[0]["x2"], faceList[0]["y2"]), (0, 0, 255), 2)
        # Image.fromarray(img[:, :, ::-1]).show()

        print(json.dumps(response))
        print("The server is still running...")
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        return jsonify(response)


if __name__ == "__main__":
    app.run(
        # host='10.136.126.13',
        host='127.0.0.1',
        port=5000
    )
