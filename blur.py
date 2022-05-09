from re import T
from turtle import forward
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from face_utils import norm_crop


SET_IMG_SHAPE = (320, 320)

"""
    Calculate smd2 with convolution
    quicker than the defination one more than 10 times
"""


@torch.no_grad()
def smd2_conv(gray, CNN: nn.Module, device: torch.device):
    gray = cv2.resize(gray, SET_IMG_SHAPE, interpolation=cv2.INTER_CUBIC)
    gray = gray / 255.0
    gray = torch.tensor(gray, dtype=torch.float32)
    gray = gray.to(device)
    smd2 = CNN(gray)
    return smd2


class SMD2_conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = np.array([[[[1.0, -1.0],  # shape: (1, 1, 2, 2)
                               [0.0, 0.0]]]])
        kernel_y = np.array([[[[1.0, 0.0],
                               [-1.0, 0.0]]]])
        self.kernel_x = nn.Parameter(torch.tensor(kernel_x, dtype=torch.float))
        self.kernel_y = nn.Parameter(torch.tensor(kernel_y, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros((1)))

    def forward(self, x):
        x = x.view(-1, 1, *x.shape)
        out_x = F.conv2d(x, self.kernel_x, self.bias)
        out_y = F.conv2d(x, self.kernel_x, self.bias)
        out_x = torch.abs(out_x)
        out_y = torch.abs(out_y)
        # out = torch.mul(out_x, out_y)
        out = out_x * out_y
        return torch.sum(out).detach().cpu()


class blur_face_dealer():
    def __init__(self, device: torch.device = None):
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.CNN = SMD2_conv2d().to(device).eval()

    def get_smd2(self, gray):
        return float(smd2_conv(gray, self.CNN, self.device))

    def blur_filter(self, img):
        smd2 = self.get_smd2(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY))
        print("smd2: ", smd2)
        if smd2 > 500:
            return True
        else:
            return False
            
def blur_filter(img):
    gray = cv2.resize(img, SET_IMG_SHAPE, interpolation=cv2.INTER_CUBIC)
    f = np.matrix(gray) / 255.0  # 返回矩阵
    x, y = f.shape
    smd2 = 0
    for i in range(x - 1):
        for j in range(y - 1):
            smd2 += np.abs(f[i, j] - f[i + 1, j]) * np.abs(f[i, j] - f[i, j + 1])
    if smd2 > 1000:
        return True
    else:
        return False

