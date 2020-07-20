import torch
from torch import nn
from torch.autograd import  Variable
import cv2
import torchvision
import numpy as np
from torchvision import  transforms
from function.GetWeights import getweights

seed = 1
torch.manual_seed(seed)

if __name__ == '__main__':
    # 所使用的原始图像载入
    img = cv2.imread('Pic/3.png')
    WR, WG, WB = getweights(img)
    print(WR,WG,WB)



