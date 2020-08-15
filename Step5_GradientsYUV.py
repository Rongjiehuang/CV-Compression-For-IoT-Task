import torch
from matplotlib import pyplot as plt
import matplotlib
import cv2
import numpy as np

seed = 1
torch.manual_seed(seed)
IMG_SHAPE = 224
from function.drn import drn_a_50
import torch
from torch import nn
from PIL import Image
import torch_dct as dct
from torchvision import transforms
from torch.autograd import Variable
from function.GetWeights import getweights
from function.RGB2YUV import RGB2YUV,sourceRGB2YUV,sourceYUV2RGB,YUV2RGB



if __name__ == '__main__':
    # 所使用的原始图像载入

    WR, WG, WB = getweights()
    img = cv2.imread('Pic/CityScapes.png')

    imgYUV = RGB2YUV(WR, WG, WB, img)
    imgYUVsource = sourceRGB2YUV(img)

    imgafter = YUV2RGB(WR, WG, WB,imgYUV)
    imgaftersource = sourceYUV2RGB(imgYUVsource)

    imgaftersource = np.uint8(imgaftersource)
    imgafter = np.uint8(imgafter)

    cv2.imshow('imgaftersource',imgaftersource)
    cv2.imshow('imgafter',imgafter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
