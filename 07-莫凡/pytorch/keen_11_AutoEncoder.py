# -*- encoding: utf-8 -*-
"""
@File    : keen_11_AutoEncoder.py
@Time    : 2019/11/19 9:55
@Author  : Keen
@Software: PyCharm
"""

import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data
import torchvision

# 超参数设置
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5


# 导入数据
train_data = torchvision.datasets.MNIST(
	root='./mnist',
	train=True,
	transform=torchvision.transforms.ToTensor(),
	target_transform=None,
	download=DOWNLOAD_MNIST)

# paint a picture
# plt.imshow(train_data.train_data[0], cmap='gray')
# plt.title("%d" % train_data.test_labels[0])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(28*28, 128),
			nn.Tanh(),
			nn.Linear(128, 64),
			nn.Tanh(),
			nn.Linear(64, 12),
			nn.Tanh(),
			nn.Linear(12, 3),
		)
		self.decoder = nn.Sequential(
			nn.Linear(3, 12),
			nn.Tanh(),
			nn.Linear(12, 64),
			nn.Tanh(),
			nn.Linear(64, 128),
			nn.Tanh(),
			nn.Linear(128, 28*28),
			nn.Sigmoid(),
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded


autoencoder = AutoEncoder()
# print(autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
	for step, (x, b_label) in enumerate(train_data):
		b_x = x.view(-1, 28*28)
		# b_y = x.view(-1, 28*28)

		encoded, decoded = autoencoder(b_x)
		# 失0反进
		loss = loss_func(decoded, b_x)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step % 100 == 0:
			print('Epoch: ', epoch, '| train loss: %.4f' % loss.data)
