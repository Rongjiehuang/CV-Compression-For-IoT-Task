import cv2
import numpy as np
from function.GetWeights import getweights
from function.dct import mydct
from function.RGB2YUV import RGB2YUV
from matplotlib import pyplot as plt
import matplotlib

if __name__ == '__main__':

    img = cv2.imread('Pic/3.png')
    img = cv2.resize(img,(224,224))
    img = img / 255

    WR, WG, WB = getweights(img)
    print(WR,WG,WB)

    #Step4 : IRGB 2 IYUV
    imgYUV = RGB2YUV(WR, WG, WB, img)

    # Step5 : IYUV 2 FYUV
    FYUV = mydct(imgYUV)

    print(FYUV.shape)
    FY = abs(FYUV[:,:,0])
    FU = abs(FYUV[:,:,1])
    FV = abs(FYUV[:,:,2])

    cmap = matplotlib.cm.jet  #### 8.绘制spectral sensitivity图  ####
    norm = matplotlib.colors.Normalize(vmin=0.0001, vmax=0.01)
    plt.imshow(FY, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.xlabel('Y Channel')
    plt.ylabel('DCT Frequency')
    plt.savefig('Pic/Figure7_Y_channel.jpg')
    plt.show()

    cmap = matplotlib.cm.jet  #### 8.绘制spectral sensitivity图  ####
    norm = matplotlib.colors.Normalize(vmin=0.0001, vmax=0.01)
    plt.imshow(FU, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.xlabel('U Channel')
    plt.ylabel('DCT Frequency')
    plt.savefig('Pic/Figure7_U_channel.jpg')
    plt.show()

    cmap = matplotlib.cm.jet  #### 8.绘制spectral sensitivity图  ####
    norm = matplotlib.colors.Normalize(vmin=0.0001, vmax=0.01)
    plt.imshow(FV, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.xlabel('V Channel')
    plt.ylabel('DCT Frequency')
    plt.savefig('Pic/Figure7_V_channel.jpg')
    plt.show()