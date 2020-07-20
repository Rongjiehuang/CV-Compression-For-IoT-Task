"""
author: muzhan
"""

import cv2
import numpy as np
from function.dct import mydct


if __name__ == '__main__':
    IRGB = cv2.imread('3.png')
    IRGB = cv2.resize(IRGB, (300, 300))
    FRGB = mydct(IRGB)
    # RGB转为HSV
    IYUV = cv2.cvtColor(IRGB,cv2.COLOR_RGB2YUV)
    FYUV = mydct(IYUV)

    FRGB = FRGB * 255
    cv2.imshow('IRGB',IRGB)
    cv2.imshow('IYUV',IYUV)
    cv2.imshow('FRGB',FRGB)
    cv2.imshow('FYUV',FYUV)

    cv2.waitKey(0)
    cv2.destroyAllWindows()