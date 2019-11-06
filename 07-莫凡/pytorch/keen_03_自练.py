# -*- encoding: utf-8 -*-
"""
@File    : keen_03_自练.py
@Time    : 2019/11/6 8:40
@Author  : Keen
@Software: PyCharm
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)

# 使用各种不同的激活函数
y_sigmoid = torch.sigmoid(x)
y_relu = F.relu(x)
y_tanh = torch.tanh(x)
y_softplus = F.softplus(x)


# 绘图
plt.figure(1)   # 长:800  宽:600    默认大小为: 640*480
plt.subplot(221)  # 两行两列，第一个
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')  # 设置图例
plt.subplot(222)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')
plt.subplot(223)
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')
plt.subplot(224)
plt.plot(x, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()