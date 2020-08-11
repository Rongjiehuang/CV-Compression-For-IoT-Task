"""
author: muzhan
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from function.dct import mydct


if __name__ == '__main__':
    IRGB = cv2.imread('Pic/CityScapes.png')

    FRGB = mydct(IRGB)
    # RGB转为HSV
    IYUV = cv2.cvtColor(IRGB,cv2.COLOR_RGB2YUV)
    FYUV = mydct(IYUV)


    FYUV_normal = abs(FYUV/FYUV.max())
    FRGB_normal = abs(FRGB/FRGB.max())


    # cv2.imshow('IRGB',IRGB)
    # cv2.imshow('IYUV',IYUV)
    # cv2.imshow('FRGB',FRGB_normal)
    # cv2.imshow('FYUV',FYUV_normal)
    # print("FYUV:", FYUV_normal.shape)
    # print("FRGB:", FRGB_normal.shape)

    cv2.imwrite('Pic/IRGB.png',IRGB)
    cv2.imwrite('Pic/IYUV.png',IYUV)
    cv2.imwrite('Pic/FRGB.png',FRGB)
    cv2.imwrite('Pic/FYUV.png',FYUV)

    # cmap = matplotlib.cm.jet  #### 8.绘制spectral sensitivity图  ####
    # norm = matplotlib.colors.Normalize(vmin=1E-6, vmax=5E-5)
    # plt.imshow(abs(FYUV_normal[:,:,1]), cmap=cmap, norm=norm)
    # plt.colorbar()
    # plt.xlabel('')
    # plt.ylabel('DCT Frequency')
    # plt.savefig('Pic/Figure5.jpg')
    # plt.show()
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()