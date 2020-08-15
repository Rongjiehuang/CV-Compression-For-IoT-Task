from function.drn import drn_a_50
import torch.utils.model_zoo as model_zoo
from PIL import Image
from torchvision import transforms
import torch
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import torch_dct as dct
from torchvision import transforms
from torch.autograd import Variable
from function.GetWeights import getweights
seed = 1
torch.manual_seed(seed)

imgxSource = Image.open("Pic/CityScapes.png").convert('RGB')
imgx = imgxSource.resize((224, 224))
tran = transforms.ToTensor()

tensorx = Variable(tran(imgx))# 1. 将图片X 转为 张量tensorX
tensors = dct.dct_3d(tensorx).view([1, 3, 224, 224])  #### 2. tensorX DCT变换得到 tensors ####
tensors = Variable(tensors, requires_grad=True)
tensorx_reverse = dct.idct_3d(tensors)  #### 3. 从tensors IDCT变换为 tensorx_reverse


criterion = nn.NLLLoss()  # 交叉熵损失函数
model = drn_a_50(True)  # 载入预训练模型Resnet50
model.eval()

out = model(tensorx_reverse).view((1,1000))  #### 4.  将tensorx_reverse 喂入DNN
print(out.shape)

target = torch.tensor([1])  ####    导入DRN模型   ####
loss = criterion(out, target)  # 计算损失
loss.backward()  #### 5. 反向传播  ####
print("tensors: ", tensors.grad.shape)  #### 6. 得到导数 ####

gR = abs(tensors.grad[0,0,:,:])
gG = abs(tensors.grad[0,1,:,:])
gB = abs(tensors.grad[0,2,:,:])



z1 = (gB / gG).flatten()
z2 = (gR / gG).flatten()
z1.sort()
z2.sort()
z1median = z1[int(len(z1) / 2)]  # 取中位数
z2median = z2[int(len(z2) / 2)]  # 取中位数

WR = z2median / (1 + z1median + z2median)  # RGB2YUV权重
WG = 1 / (1 + z1median + z2median)
WB = z1median / (1 + z1median + z2median)

WR=WR.item()
WG=WG.item()
WB=WB.item()

plt.bar(x=1, height=WG, width=0.1, color='black', label="Y", bottom=0)
plt.bar(x=1, height=WB, width=0.1, color='blue', label="U", bottom=WG)
plt.bar(x=1, height=WR, width=0.1, color='red', label="V", bottom=WG + WB)

plt.bar(x=1.5, height=0.587, width=0.1, color='black', bottom=0)
plt.bar(x=1.5, height=0.299, width=0.1, color='blue', bottom=0.587)
plt.bar(x=1.5, height=0.114, width=0.1, color='red', bottom=0.299 + 0.587)

plt.xticks([1, 1.5], ["G-YUV", "YUV"])
plt.ylim(0, 1)
plt.ylabel("Sensitivity")
plt.xlabel("")
plt.title("Target:Resnet50")

plt.legend()
plt.savefig('Pic/Figure10.jpg')
plt.show()