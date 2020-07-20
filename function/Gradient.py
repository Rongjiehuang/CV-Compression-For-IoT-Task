import torch
from torch import nn
from torchviz import make_dot
from torch.autograd import  Variable
import cv2
import torchvision
import numpy as np
from torchvision import  transforms
import sys
from function.dct import mydct

# model = torch.load("..\DRN\drn-master\model\drn_d_22_cityscapes.pth", map_location=torch.device('cpu'))

seed = 1
torch.manual_seed(seed)

def mysolve(imgx):  #单图像导数计算

    imgs = mydct(imgx)  #### 2.DCT变换得到初始DCT系数s ####

    tran = transforms.ToTensor()
    imgs = cv2.resize(imgs, (224, 224))

    tensorS = tran(imgs)  # 将图片s数组转为张量 tensorS
    tensorS = Variable(tensorS, requires_grad=True)

    Rs = Variable(tensorS[0], requires_grad=True)  # 取出tensorS的R、G、B三个通道信息
    Gs = Variable(tensorS[1], requires_grad=True)
    Bs = Variable(tensorS[2], requires_grad=True)
    Rs.retain_grad()  # 保存导数
    Gs.retain_grad()
    Bs.retain_grad()

    N = m = n = 224  #### 3.IDCT变换 从s变为x  ####
    C = np.zeros((m, n))
    C[0, :] = 1 * np.sqrt(1 / N)
    for i in range(1, m):
        for j in range(n):
            C[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)) * np.sqrt(2 / N)  # 得到IDCT变换所需的系数矩阵

    C = tran(C)
    C = Variable(C, requires_grad=True).view([224, 224])
    C_T = Variable(torch.t(C), requires_grad=True)

    Rtemp = C_T * Rs  # IDCT变换公式： f = C^T * F * C
    R = Rtemp * C
    Gtemp = C_T * Gs
    G = Gtemp * C
    Btemp = C_T * Bs
    B = Btemp * C

    temp = torch.stack([R, G, B], dim=0).view([1, 3, 224, 224])  # 将IDCT变换后的RGB分量拼接成x
    temp.retain_grad()  # 保存导数信息
    tensorX = temp.float()

    criterion = nn.NLLLoss()  # 交叉熵损失函数
    model = torchvision.models.resnet50()  #### 4.导入ResNet50模型  ####

    model.eval()
    out = model(tensorX)  # 输出结果：1000分类的信息 shape：[1,1000]
    target = torch.tensor([0])  #### 5.输出结果 由于使用的Resnet50为1000分类问题，这里假设分类标签为[0]  ####
    loss = criterion(out, target)  # 计算损失

    loss.backward()  #### 6.反向传播  ####

    print("backward: ", loss)
    gradR = abs(Rs.grad)  #### 7.得到s的三通道RGB导数信息  ####
    gradG = abs(Gs.grad)
    gradB = abs(Bs.grad)

    return np.array(gradR),np.array(gradG),np.array(gradB)
