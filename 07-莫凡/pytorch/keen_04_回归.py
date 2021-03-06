# -*- encoding: utf-8 -*-
"""
@File    : keen_04_回归.py
@Time    : 2019/10/22 9:59
@Author  : Keen
@Software: PyCharm
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 本来是一维的，后来加了一个维度
x = torch.linspace(-1, 1, 100)
print(x.size())
x = torch.unsqueeze(x, dim=1)  # 每个列元素，加一个维度
print(x.size())
# 1:从上往下加维度 shape: [100, 1]    0:从左往右加维度 shape: [1,100] # y = x^2 + 高斯白噪声
y = x.pow(2) + 0.2 * torch.rand(x.size())


class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
		self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

	def forward(self, x):
		x = F.relu(self.hidden(x))  # activation function for hidden layer
		x = self.predict(x)  # linear output
		return x


net = Net(n_feature=1, n_hidden=10, n_output=1)  # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# plt.scatter(x, y)
# plt.plot(x, prediction, 'r-', lw=5)

# plt.ion()  # 开启了交互模式 与plt.ioff() 对应，意为关闭交互模式

for t in range(200):
	prediction = net(x)  # input x and predict based on x
	loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)
	optimizer.zero_grad()  # clear gradients for next train
	loss.backward()  # backpropagation, compute gradients
	optimizer.step()  # apply gradients
	print(prediction.size())  # torch.Size([100, 1])
	if t % 20 == 0:
		# plot and show learning process
		plt.cla()
		plt.scatter(x.data.numpy(), y.data.numpy())
		plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
		plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
		plt.pause(0.01)
		print(t)
# plt.ioff()
plt.show()
