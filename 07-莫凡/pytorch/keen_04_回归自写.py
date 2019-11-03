# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# 本来是一维的，后来加了一个维度
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 1:从上往下加维度 shape: [100, 1]    0:从左往右加维度 shape: [1,100]
# y = x^2 + 高斯白噪声
y = x.pow(2) + 0.2 * torch.rand(x.size())

print(y)
# 画散点图
# plt.scatter(x, y)
# plt.show()


# 建立神经网路
class Net(nn.Module):

	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.hidden = nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
		self.predict = nn.Linear(n_hidden, n_output)  # 输出层线性输出

	# 重写了module中的forward
	def forward(self, x):
		x = F.relu(self.hidden(x))
		x = self.predict(x)
		return x


net = Net(n_feature=1, n_hidden=10, n_output=1)

# 显示网络结构
print(net)

# 训练网络
# optimizer为训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入net的所有参数，学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值之间的误差计算公式(MSE)   均方差损失

# 可视化训练过程
plt.ion()  # 画图

for t in range(150):  # 迭代对数据进行训练

	prediction = net(x)  # input x and predict based on x

	loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)   计算两者间的误差

	optimizer.zero_grad()  # clear gradients for next train  清空上一步的残余更新参数
	loss.backward()  # backpropagation, compute gradients  误差反向传播，计算参数更新值
	optimizer.step()  # apply gradients  将参数更新值施加到net的paramters上，即对参数进行优化

	if t % 5 == 0:
		# plot and show learning process
		plt.cla()
		plt.scatter(x, y)
		plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
		plt.text(0.5, 0, 'Loss=%.4f' % loss, fontdict={'size': 20, 'color': 'red'})
		plt.pause(0.1)

plt.ioff()
plt.show()