import torch
from matplotlib import pyplot as plt
import matplotlib
import cv2

import numpy as np

from function.GetWeights import getweights
from function.dct import mydct
from function.RGB2YUV import RGB2YUV
seed = 1
torch.manual_seed(seed)

if __name__ == '__main__':
    # 所使用的原始图像载入

    img = cv2.imread('Pic/3.png')
    img = cv2.resize(img,(224,224))
    img = img / 255

    WR, WG, WB = getweights(img)
    print(WR,WG,WB)

    #Step4 : IRGB 2 IYUV
    imgYUV = RGB2YUV(WR, WG, WB, img)

    # Step5 : IYUV 2 FYUV
    FYUV = mydct(imgYUV)

    FY = abs(FYUV[:,:,0])
    FU = abs(FYUV[:,:,1])
    FV = abs(FYUV[:,:,2])

    Usensitivity = np.array(FU).sum()/(224*224)
    Vsensitivity = np.array(FV).sum()/(224*224)

    print("Usensitivity",Usensitivity)
    print("Vsensitivity",Vsensitivity)


    rects1 = plt.bar(x=1,height=Usensitivity, width=0.1, color='yellow', label="Usensitivity")
    rects2 = plt.bar(x=1,height=Vsensitivity, width=0.1, color='green', label="Vsensitivity")
    rects0 = plt.bar(x=1,height=0.2-Usensitivity-Vsensitivity, width=0.1, color='red', label="Usensitivity")
    plt.ylim(0, 1)
    plt.ylabel("Sensitivity")
    plt.xlabel("GYUV")
    plt.title("Target:Resnet50")
    plt.legend()
    plt.show()
