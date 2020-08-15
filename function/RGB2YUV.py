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

def sourceRGB2YUV(img):
    WR = 0.299
    WG = 0.587
    WB = 0.114
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

def sourceYUV2RGB(img):
    WR = 0.299
    WG = 0.587
    WB = 0.114
    (y, u, v) = cv2.split(img)
    r = y + 1.44*v
    g = y - 0.39*u - 0.58*v
    b = y + 2.03*u

    imgRGB = cv2.merge([r, g, b])  # IDCT复原图像
    return imgRGB

def YUV2RGB(WR,WG,WB,img):
    (y, u, v) = cv2.split(img)
    b = y + 2*(1-WB)*u
    r = y + 2*(1-WR)*v
    g = (y - (WR*r + WB*b))/WG
    imgRGB = cv2.merge([r, g, b])  # IDCT复原图像
    return imgRGB