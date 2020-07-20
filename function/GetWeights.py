
from function.Gradient import mysolve

def getweights(img):
    gR, gG, gB = mysolve(img)
    z1 = (gB / gG).flatten()
    z2 = (gR / gG).flatten()
    z1.sort()
    z2.sort()
    z1median = z1[int(len(z1) / 2)]  # 取中位数
    z2median = z2[int(len(z2) / 2)]  # 取中位数

    WR = z2median/(1+z1median+z2median)   #RGB2YUV权重
    WG = 1/(1+z1median+z2median)
    WB = z1median/(1+z1median+z2median)

    return WR,WG,WB