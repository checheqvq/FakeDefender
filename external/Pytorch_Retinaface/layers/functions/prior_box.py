import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        # ceil() 向上取整，舍小数，正数部分进1 （若为负，则仅舍小数）
        # print(self.feature_maps) --> [[54,96],[27,48],[14,24]]
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    # 获取一系列默认框
    def forward(self):
        anchors = []
        # 遍历三个特征图，获取默认框大小 min_sizes[k]
        # enumerate() 返回 下标号，元素
        # print(f) --> [54,96] [27,48] [14,24]
        # print(k) --> 0 1 2
        for k, f in enumerate(self.feature_maps):
            # 第一次循环 min_sizes --> [16,32] 第二次 --> [64,128] 第三次 --> [256,512]
            min_sizes = self.min_sizes[k]
            # 遍历特征图的每个像素点  product()返回相应的笛卡尔乘积
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    # s_kx 默认框的宽w s_ky 默认框的高h dense_cx,dense_cy 默认框的中心点
                    # image_size --> (height, width)
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    # x,y乘以step/image_size 因为feature_map和steps的乘积正好是image的大小
                    # steps --> [8,16,32]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    # 变化anchors的形状[x,y,w,h]，并将其归一化于0-1之间
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        # self.clip --> false  并不执行
        if self.clip:
            # output.clamp_ 把张量output的每个元素的值压缩到区间[min, max]之间，并且修改后直接返回给output，其中小于min的记为min，大于max的记为max
            output.clamp_(max=1, min=0)
        return output
