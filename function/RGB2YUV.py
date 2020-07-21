import cv2
import numpy as np

def RGB2YUV(WR,WG,WB,img):

    (r, g, b) = cv2.split(img)
    y = WR * r + WG * g + WB * b
    u = 1/2 * (b-y) / (1-WB)
    v = 1/2 * (r-y) / (1-WR)

    # y = np.array(y).astype(np.int8)
    # u = np.array(u).astype(np.int8)
    # v = np.array(v).astype(np.int8)
    imgYUV = cv2.merge([y, u, v])  # IDCT复原图像
    # imgcvYUV = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

    return imgYUV