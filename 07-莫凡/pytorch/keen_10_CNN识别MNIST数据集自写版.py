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
	# mnist文件夹不存在 或者 mnist文件夹中为空，则下载mnist数据集
	DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST( # 返回一个tuple，可以使用索引来访问
	root="./mnist/",
	train=True,
	transform=torchvision.transforms.ToTensor(),

	download=DOWNLOAD_MNIST,
)

# 展示一个图片的例子
# print(train_data.train_data.size())  # train_data  torch.Size([60000, 28, 28]) 60000张28*28的图片
# print(train_data.train_labels.size())  # train_labels torch.Size([60000])
# plt.imshow(train_data.train_data[0], cmap="gray")
# plt.title("%i" % train_data.train_labels[0])
# plt.show()


# 将训练数据集装载入DataLoader
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 选取2000张图片作为训练数据集
test_data = torchvision.datasets.MNIST(root="./mnist/", train=False)
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]
print(test_x[0].size())
print(test_x[0])
plt.imshow(test_data.test_data[0], cmap="gray")
plt.title("%d" % test_data.test_labels[0])
print(test_data.test_labels[0])
plt.show()


# 构建CNN网络
class CNN(nn.Module):
	def __init__(self):
		# 代表哪个子类， 并将自身传入
		# super(CNN, self).__init__()
		super().__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=1,
				out_channels=16,
				kernel_size=5, # filter size    5*5
				stride=1,
				padding=2,  # padding=(kernel_size-1)/2 if stride=1
			),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, 5, 1, 2),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # shape：   =》 new_shape: (batch_size, 32 * 7 * 7)
		output = self.out(x)
		return output,x


cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()




