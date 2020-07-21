import torch
from torch import nn
from torch.autograd import Variable
import cv2
import torchvision
from scipy import fftpack
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from torchvision import transforms


def mydct(img):
    # DCT变换
    (r, g, b) = cv2.split(img)

    imgb = np.float32(b)  # 将数值精度调整为32位浮点型
    imgg = np.float32(g)  # 将数值精度调整为32位浮点型
    imgr = np.float32(r)  # 将数值精度调整为32位浮点型

    imgb_dct = cv2.dct(imgb)  # 使用dct获得img的频域图像
    imgg_dct = cv2.dct(imgg)  # 使用dct获得img的频域图像
    imgr_dct = cv2.dct(imgr)  # 使用dct获得img的频域图像

    imgb_idct = cv2.idct(imgb_dct)  # 使用反dct从频域图像恢复出原图像(有损)
    imgg_idct = cv2.idct(imgg_dct)  # 使用反dct从频域图像恢复出原图像(有损)
    imgr_idct = cv2.idct(imgr_dct)  # 使用反dct从频域图像恢复出原图像(有损)

    dstb = np.zeros(imgb_idct.shape, dtype=np.float32)
    dstg = np.zeros(imgg_idct.shape, dtype=np.float32)
    dstr = np.zeros(imgr_idct.shape, dtype=np.float32)

    # cv2.normalize(imgb_idct, dst=dstb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # cv2.normalize(imgg_idct, dst=dstg, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # cv2.normalize(imgr_idct, dst=dstr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    source = cv2.merge([r, g, b])
    result_dct = cv2.merge([imgr_dct, imgg_dct, imgb_dct])
    result_idct = cv2.merge([dstr, dstg, dstb])

    return result_dct
