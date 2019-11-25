# -*- encoding: utf-8 -*-
"""
@File    : keen_00_test.py
@Time    : 2019/10/29 9:30
@Author  : Keen
@Software: PyCharm
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
# TODO  test: 测试
# TODO  test1: 测试维度添加
# x = torch.Tensor([[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 8, 9]])
# print(x, x.shape)
# x = torch.unsqueeze(x, dim=0)  # 水平添加维度，即最外围加[]提升一维
# print(x, x.shape)  # torch.Size([1, 3, 3])   有1个元素，每个元素是3行3列
#
# print(x, x.shape)  # torch.Size([3, 3]
# x = torch.unsqueeze(x, dim=1)  # 竖直方向添加维度，即起始内部每个元素都加一个[]
# print(x, x.shape)  # torch.Size([3, 1, 3])  有3个元素，每个元素1行3列

# TODO  test2: 测试torch.rand
# y = torch.rand(5)  # 返回在[0,1]区间内的5个随机数 组成的张量
# print(y)

# TODO  test3: 测试torch.ones(5,2)    二维tensor
# n_data = torch.ones(5, 2)  # 5行2列 的二维tensor
# print(n_data)
# print(n_data.shape)

# TODO  test4: 测试torch.normal(means= ,std= ,out=)    和test3一同测试
# x0 = torch.normal(2*n_data, 1)  # 以[2, 2]为均值，标准差为1的正态分布中随机取值
# print(x0)

# TODO test5：测试tensor.view(a, b)   将tensor转换为a行，b列，当b=-1时，代表所有元素÷a行 = b
x = torch.ones(2, 3)
x[1][2] = 2
# tensor([[1., 1., 1.],
#         [1., 1., 2.]])
# x = x.view(3, -1)
# print(x)
# print(x.view(-1, 2, 3)) # -1所在位会自动进行推断
# tensor([[1., 1.],
#         [1., 1.],
#         [1., 2.]])

# TODO test: 测试matplotlib画图
# x = torch.linspace(-1, 1, 100)
# y = x.pow(2)
# plt.scatter(x, y)
# plt.show()
