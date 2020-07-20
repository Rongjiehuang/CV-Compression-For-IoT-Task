import torch
from torch import nn
from torchviz import make_dot
from torch.autograd import  Variable
import cv2
import torchvision
import numpy as np
from torchvision import  transforms
import sys
from function.Gradient import mysolve

# model = torch.load("..\DRN\drn-master\model\drn_d_22_cityscapes.pth", map_location=torch.device('cpu'))

seed = 1
torch.manual_seed(seed)

if __name__ == '__main__':
    # 所使用的原始图像载入
    img = cv2.imread('Pic/3.png')  #### 1.读入图片 ####
    gR,gG,gB = mysolve(img)
    print(gR,gG,gB)



