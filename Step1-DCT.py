"""
author: muzhan
"""

import cv2
import numpy as np


def mydct(img):
# DCT变换
    (r, g, b)= cv2.split(img)

    imgb = np.float32(b)  # 将数值精度调整为32位浮点型
    imgg = np.float32(g)  # 将数值精度调整为32位浮点型
    imgr = np.float32(r)  # 将数值精度调整为32位浮点型

    imgb_dct = cv2.dct(imgb)  # 使用dct获得img的频域图像
    imgg_dct = cv2.dct(imgg)  # 使用dct获得img的频域图像
    imgr_dct = cv2.dct(imgr)  # 使用dct获得img的频域图像

    imgb_idct = cv2.idct(imgb_dct)  # 使用反dct从频域图像恢复出原图像(有损)
    imgg_idct = cv2.idct(imgg_dct)  # 使用反dct从频域图像恢复出原图像(有损)
    imgr_idct = cv2.idct(imgr_dct)  # 使用反dct从频域图像恢复出原图像(有损)

    dstb = np.zeros(imgb_idct.shape,dtype=np.float32)
    dstg = np.zeros(imgg_idct.shape,dtype=np.float32)
    dstr = np.zeros(imgr_idct.shape,dtype=np.float32)

    cv2.normalize(imgb_idct,dst=dstb,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    cv2.normalize(imgg_idct,dst=dstg,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    cv2.normalize(imgr_idct,dst=dstr,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)

    source = cv2.merge([r,g,b])   #原图
    result_dct = cv2.merge([imgr_dct,imgg_dct,imgb_dct])  #DCT变换后图像
    result_idct = cv2.merge([dstr,dstg,dstb])  #IDCT复原图像

    return result_dct

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