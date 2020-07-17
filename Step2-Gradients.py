import torch
from torch import nn
from torchviz import make_dot
from torch.autograd import  Variable
import cv2
import torchvision
import numpy as np
from torchvision import  transforms
# model = torch.load("..\DRN\drn-master\model\drn_d_22_cityscapes.pth", map_location=torch.device('cpu'))

seed = 1
torch.manual_seed(seed)

def mysolve(img1):  #单图像导数计算
    tran = transforms.ToTensor()
    img1 = cv2.resize(img1,(224,224))
    # Tensor大小转换
    img_tensor1 = tran(img1)
    img_tensor1 = img_tensor1.view([1, 3, 224, 224])
    # Label（1000类）
    label = torch.ones((1,1000))

    criterion = nn.CrossEntropyLoss()  #核心：交叉熵损失函数
    model = torchvision.models.resnet50()  #导入ResNet50模型
    img_tensor1 = Variable(img_tensor1,requires_grad=True)
    # label = Variable(label,requires_grad=True)
    model.eval()

    out1 = model(img_tensor1)   #输出结果：1000分类的信息 shape：[1,1000]
    target = torch.tensor([0])  #shape：1000分类标签 shape:1
    loss = criterion(out1,target)   #计算损失
    print("theloss:",loss)

    loss.backward()                 #反向传播
    print("backward:",loss)
    print("grad:",img_tensor1.grad.data.shape)  #计算导数信息

    return loss,img_tensor1.grad


def mulitsolve():  #多图像数据集的导数计算
    # 所使用的原始图像载入
    img = cv2.imread('3.png')
    result = []
    for i in range(5):
        result.append(img)

    totalloss = []
    totalgrad = []
    for im in result:
        loss, grad= mysolve(im)
        totalloss.append(loss)
        totalgrad.append(grad)
    print(totalgrad)
    print(totalloss)

if __name__ == '__main__':
    # 所使用的原始图像载入
    img = cv2.imread('3.png')
    # (r, g, b) = cv2.split(img)
    loss,grad = mysolve(img)
    grad = np.array(grad)

    gR = grad[0,0,:,:]
    gG = grad[0,1,:,:]
    gB = grad[0,2,:,:]

    z1 = (gB/gG).flatten()
    z2 = (gR/gG).flatten()
    z1.sort()
    z2.sort()
    z1median = z1[int(len(z1)/2)]
    z2median = z2[int(len(z2)/2)]

    WR = z2median/(1+z1median+z2median)
    WG = 1/(1+z1median+z2median)
    WB = z1median/(1+z1median+z2median)

    print(WR,WG,WB)



