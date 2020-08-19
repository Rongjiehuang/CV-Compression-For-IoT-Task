import cv2
import numpy as np

def sourceRGB2YUV(img,WR=0.299,WG=0.587,WB=0.114): #R，G，B~[0,255]   U，V~[-128,128]
    img = img / 255

    #source:
    #WR = 0.299
    # WG = 0.587
    # WB = 0.114

    (b, g, r) = cv2.split(img)

    y = WR * r + WG * g + WB * b
    u = 1/2 * (b-y) / (1-WB)  # Cb
    v = 1/2 * (r-y) / (1-WR)  # Cr

    # 量化前  Y~ [0,1]     U,V~[-0.5,0.5]
    # 量化后  Y~[16,235]   U ~[16,240]   V~[16,240]
    y = y*(235-16) +16
    u = u*224 +128
    v = v*224 +128

    y = y.astype(np.int)
    u = u.astype(np.int)
    v = v.astype(np.int)

    imgYUV = cv2.merge([y,u,v])
    return imgYUV

def sourceYUV2RGB(img,WR=0.299,WG=0.587,WB=0.114): #R，G，B~[0,255]   U，V~[-128,128]
    # 量化前   Y~[16,235]   U ~[16,240]   V~[16,240]
    # 量化后   Y~ [0,1]     U,V~[-0.5,0.5]
    (v,u,y) = cv2.split(img)
    y = 1/219 *y - 16/219
    u = 1/224 *u - 0.5714
    v = 1/224 *v - 0.5714

    # print("y:", y, "u:", u, "v:", v)

    b = y + 2*(1-WB)*u
    r = y + 2*(1-WR)*v
    g = (y - (WR*r + WB*b))/WG
    # print("r:", r, "g:", g, "b:", b)
    imgRGB = cv2.merge([r,g,b])
    imgRGB = imgRGB*255
    # imgRGB.astype(np.int)
    return imgRGB
