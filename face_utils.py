import time

import cv2
import numpy as np

from torch import torch
# skimage是Scikit-Image 基于python脚本语言开发的数字图片处理包
# 它将图片作为numpy数组进行处理，正好与matlab一样
# transform模块的主要实现功能是 几何变换或其他变换，如旋转、拉伸和拉东变换等
# SimilarityTransform 相似变换
from skimage.transform import SimilarityTransform
from external.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from external.Pytorch_Retinaface.data import cfg_re50
from external.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from external.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from external.Pytorch_Retinaface.models.retinaface import RetinaFace


def norm_crop(img, landmark, image_size=112):
    ARCFACE_SRC = np.array([[
        [122.5, 141.25],
        [197.5, 141.25],
        [160.0, 178.75],
        [137.5, 225.25],
        [182.5, 225.25]
    ]], dtype=np.float32)

    def estimate_norm(lmk):
        assert lmk.shape == (5, 2)

        tform = SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = np.inf
        src = ARCFACE_SRC

        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]

        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))

        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

        return min_M, min_index

    M, pose_index = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


class FaceDetector:
    # cofidence_threshold 置信阈值
    def __init__(self, device="cpu", confidence_threshold=0.8):
        self.device = device
        self.confidence_threshold = confidence_threshold

        # cfg configuration 配置/结构
        self.cfg = cfg = cfg_re50
        # variance 方差n
        self.variance = cfg["variance"]
        cfg["pretrain"] = False

        self.net = RetinaFace(cfg=cfg, phase="test").to(device).eval()
        self.decode_param_cache = {}

    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))

    def decode_params(self, height, width):
        cache_key = (height, width)

        try:
            return self.decode_param_cache[cache_key]
        except KeyError:
            priorbox = PriorBox(self.cfg, image_size=(height, width))
            priors = priorbox.forward()

            # priors 返回一个tensor tensor.data返回和priors相同的数据，但是不会加入到priors的计算历史里
            prior_data = priors.data
            # torch.Tensor([width, height] * 2) --> tensor([768., 432., 768., 432.])
            scale = torch.Tensor([width, height] * 2)
            scale1 = torch.Tensor([width, height] * 5)

            result = (prior_data, scale, scale1)
            self.decode_param_cache[cache_key] = result
            return result

    def detect(self, img):
        device = self.device

        prior_data, scale, scale1 = self.decode_params(*img.shape[:2])

        # REF: test_fddb.py
        # float32() 在内存中占32个位
        img = np.float32(img)
        img -= (104, 117, 123)
        # transpose 调换数组索引值，x y z --> z x y
        img = img.transpose(2, 0, 1)
        # from_numpy 把数组转换成张量 unsqueeze(0) 在第一维，增加一个维度
        img = torch.from_numpy(img).unsqueeze(0)
        # to() 使用指定的device和dtype返回张量
        img = img.to(device, dtype=torch.float32)

        loc, conf, landms = self.net(img)
        # tensor.cpu()   转为 CPU tensor
        loc = loc.cpu()
        conf = conf.cpu()
        landms = landms.cpu()


        # Decode results
        boxes = decode(loc.data.squeeze(0), prior_data, self.variance)
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # decode_landm 从预测中解读landms，使用priors来撤销我们在训练时对偏移回归所做的编码
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.variance)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]     # 获取满足阈值的人脸框和关键点
        landms = landms[inds]
        scores = scores[inds]

        top_k = 5000
        nms_threshold = 0.4
        keep_top_k = 750
        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        boxes = dets[:keep_top_k, :][:, :4]
        landms = landms[:keep_top_k, :]

        return boxes, landms
