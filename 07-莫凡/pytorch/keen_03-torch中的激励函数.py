# -*- encoding: utf-8 -*-
"""
@File    : keen_03-torch中的激励函数.py
@Time    : 2019/10/21 16:03
@Author  : Keen
@Software: PyCharm
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

x = torch.linspace(-5, 5, 200)  # -5 -> 5 等分为200份
x_np = x.numpy()
# x = Variable(x) # x节点可以被反向传播    在现在版本中已经可以不使用了
# print(type(x))  # <class 'torch.Tensor'>
# print(x)

# 生成不同的激励函数数据
y_relu = torch.relu(x).numpy()
y_sigmoid = torch.sigmoid(x).numpy()
y_tanh = torch.tanh(x).numpy()
y_softplus = F.softplus(x)
# y_softmax = F.softmax(x)

# 绘图
plt.figure(1)   # 长:800  宽:600    默认大小为: 640*480
plt.subplot(221)  # 两行两列，第一个
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')  # 设置图例
plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')
plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')
plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
