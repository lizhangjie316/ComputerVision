# -*- encoding: utf-8 -*-
"""
@File    : keen_10_CNN识别MNIST数据集自写版.py
@Time    : 2019/11/8 16:27
@Author  : Keen
@Software: PyCharm
"""
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)

# 超参设置
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# 处理数据
if not(os.path.exists("./mnist/")) or not(os.listdir("./mnist/")):
	DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
	root="./mnist/",
	train=True,
	transform=torchvision.transforms.ToTensor(),

	download=DOWNLOAD_MNIST,
)

print(train_data.data.size())  # train_data  torch.Size([60000, 28, 28])
print(train_data.targets.size())  # train_labels torch.Size([60000])
plt.imshow(train_data.data[0], cmap="gray")
plt.title("%i" % train_data.targets[0])



