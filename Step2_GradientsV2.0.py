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
import matplotlib
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

gradR = abs(tensors.grad[0,2,:,:])

# print("gradR: ", gradR)  #### 6. 得到导数 ####
cmap = matplotlib.cm.jet  #### 8.绘制spectral sensitivity图  ####
norm = matplotlib.colors.Normalize(vmin=1E-8, vmax=1E-5)
plt.imshow(gradR, cmap=cmap, norm=norm)
plt.colorbar()
plt.xlabel('B Channel')
plt.ylabel('DCT Frequency')
plt.savefig('Pic/Figure7_B_channel.jpg')
plt.show()

