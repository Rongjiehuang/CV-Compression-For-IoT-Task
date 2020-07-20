import cv2
import numpy as np
from function.GetWeights import getweights
from function.dct import mydct

def RGB2YUV(WR,WG,WB,img):

    (r, g, b) = cv2.split(img)
    y = WR * r + WG * g + WB * b
    u = 1/2 * (b-y) / (1-WB)
    v = 1/2 * (r-y) / (1-WR)

    y = np.array(y).astype(np.int8)
    u = np.array(u).astype(np.int8)
    v = np.array(v).astype(np.int8)
    imgYUV = cv2.merge([y, u, v])  # IDCT复原图像
    imgcvYUV = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

    return imgYUV,imgcvYUV

if __name__ == '__main__':

    img = cv2.imread('Pic/3.png')
    WR, WG, WB = getweights(img)

    #Step4 : IRGB 2 IYUV
    imgYUV,imgcvYUV = RGB2YUV(WR, WG, WB, img)


    # Step5 : IYUV 2 FYUV
    FYUV = mydct(imgYUV)

    # cv2.imshow('imgYUV', imgYUV)
    print(FYUV)
    cv2.imshow('imgcvYUV', FYUV)
    cv2.waitKey(0)
    cv2.destroyAllWindows()