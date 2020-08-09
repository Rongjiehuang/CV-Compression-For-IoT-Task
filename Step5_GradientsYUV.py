import torch
from matplotlib import pyplot as plt
import matplotlib
import cv2

import numpy as np
from function.Gradient import mysolve
from function.GetWeights import getweights
from function.dct import mydct
from function.RGB2YUV import RGB2YUV
seed = 1
torch.manual_seed(seed)
IMG_SHAPE = 224

if __name__ == '__main__':
    # 所使用的原始图像载入

    img = cv2.imread('Pic/3.png')
    img = cv2.resize(img,(IMG_SHAPE,IMG_SHAPE))
    img = img / 255

    WR, WG, WB = getweights(img)
    print(WR,WG,WB)

    #Step4 : IRGB 2 IYUV
    imgYUV = RGB2YUV(WR, WG, WB, img)

    # Step5 : IYUV 2 FYUV
    FYUV = mydct(imgYUV)
    gY, gU, gV = mysolve(FYUV)

    # print("gR",gY)
    # print("gG",gU)
    # print("gB",gV)

    gYvalue = np.array(gY).sum()/(IMG_SHAPE*IMG_SHAPE)
    gUvalue = np.array(gU).sum()/(IMG_SHAPE*IMG_SHAPE)
    gVvalue = np.array(gV).sum()/(IMG_SHAPE*IMG_SHAPE)


    print("gYvalue",gYvalue)
    print("gUvalue",gUvalue)
    print("gVvalue",gVvalue)

    #归一化
    gYvalue_normal = gYvalue/(gYvalue+gUvalue+gVvalue)
    gUvalue_normal = gUvalue/(gYvalue+gUvalue+gVvalue)
    gVvalue_normal = gVvalue/(gYvalue+gUvalue+gVvalue)


    print("gYvalue_normal",gYvalue_normal)
    print("gUvalue_normal",gUvalue_normal)
    print("gVvalue_normal",gVvalue_normal)


    plt.bar(x=1,height=gYvalue_normal, width=0.1, color='black', label="Y",bottom=0)
    plt.bar(x=1,height=gUvalue_normal, width=0.1, color='blue', label="Y",bottom=gYvalue_normal)
    plt.bar(x=1,height=gVvalue_normal, width=0.1, color='red', label="V",bottom=gYvalue_normal+gVvalue_normal)
    # plt.xticks([1, 1.5], ["Human", "DNN"])
    plt.ylim(0, 1)
    plt.ylabel("Sensitivity")
    plt.xlabel("GYUV")
    plt.title("Target:Resnet50")

    plt.legend()
    # plt.savefig('Pic/Figure6_ColorSensitivity.jpg')
    plt.show()
