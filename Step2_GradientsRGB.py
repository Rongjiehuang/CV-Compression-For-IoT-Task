import torch
from torch import nn
from torchviz import make_dot
from torch.autograd import  Variable
import cv2
import torchvision
import numpy as np
from torchvision import  transforms
import sys
from matplotlib import pyplot as plt
import matplotlib
from function.Gradient import mysolve

# model = torch.load("..\DRN\drn-master\model\drn_d_22_cityscapes.pth", map_location=torch.device('cpu'))

seed = 1
torch.manual_seed(seed)

if __name__ == '__main__':
    # 所使用的原始图像载入
    img = cv2.imread('Pic/3.png')  #### 1.读入图片 ####
    gR,gG,gB = mysolve(img)
    # print(gR,gG,gB)

    gRvalue = np.array(gR).sum()
    gGvalue = np.array(gG).sum()
    gBvalue = np.array(gB).sum()

    print("gR",gRvalue)
    print("gG",gGvalue)
    print("gB",gBvalue)


    plt.bar(x=1,height=gRvalue, width=0.1, color='red', label="R",bottom=0)
    plt.bar(x=1,height=gGvalue, width=0.1, color='green', label="G",bottom=gRvalue)
    plt.bar(x=1,height=gBvalue, width=0.1, color='blue', label="B",bottom=gRvalue+gGvalue)

    plt.bar(x=1.5, height=0.299, width=0.1, color='red', bottom=0)
    plt.bar(x=1.5, height=0.587, width=0.1, color='green', bottom=0.299)
    plt.bar(x=1.5, height=0.114, width=0.1, color='blue', bottom=0.299 + 0.587)

    plt.xticks([1,1.5],["Human","DNN"])
    plt.ylim(0,1)
    plt.ylabel("Sensitivity")
    plt.xlabel("GRGB")
    plt.title("Target:Resnet50")

    plt.legend()
    # plt.savefig('Pic/Figure6_ColorSensitivity.jpg')
    plt.show()




