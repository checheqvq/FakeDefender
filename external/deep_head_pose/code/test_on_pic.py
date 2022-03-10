import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, utils

if __name__ == '__main__':
    img = cv2.imread('D:/windows_data/desktop/test/frontal_2.jpg', 1)
    bbox_line_list = [[ 90.824684,40.590965,262.55737,261.0923]]

    cudnn.enabled = True

    batch_size = 1
    gpu = 0
    snapshot_path = '../model/hopenet_robust_alpha1.pkl'

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    # New cv2
    height, width, channels = img.shape

    cv2_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for bbox in bbox_line_list:
        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)
        # x_min -= 3 * bbox_width / 4
        # x_max += 3 * bbox_width / 4
        # y_min -= 3 * bbox_height / 4
        # y_max += bbox_height / 4
        x_min -= 50
        x_max += 50
        y_min -= 50
        y_max += 30
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        # Crop image
        img = cv2_frame[y_min:y_max,x_min:x_max]
        img = Image.fromarray(img)
        img.show()

        # Transform
        img = transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda(gpu)

        yaw, pitch, roll = model(img)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        print(' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))