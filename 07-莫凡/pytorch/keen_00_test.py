# -*- encoding: utf-8 -*-
"""
@File    : keen_00_test.py
@Time    : 2019/10/29 9:30
@Author  : Keen
@Software: PyCharm
"""

import torch
import numpy as np
# TODO  test: 测试
# TODO  test1: 测试维度添加
# x = torch.from_numpy(np.array([[1, 2, 3],
#                                [4, 5, 6],
#                                [7, 8, 9]]))
# # x = torch.unsqueeze(x, dim=0)  # 水平添加维度，即最外围加[]提升一维
# # print(x, x.shape)  # torch.Size([1, 3, 3])   有1个元素，每个元素是3行3列
#
# x = torch.unsqueeze(x, dim=1)  # 竖直方向添加维度，即起始内部每个元素都加一个[]
# print(x, x.shape)  # torch.Size([3, 1, 3])  有3个元素，每个元素1行3列

# TODO  test2: 测试torch.rand
# y = torch.rand(100) # 返回在[0,1]区间内均匀分布的100个随机数 组成的张量
# print(y)

# TODO  test3: 测试torch.ones(10,2)    二维tensor
n_data = torch.ones(10, 2)  # 10行2列 的二维tensor
# print(n_data)
# print(n_data.shape)

# TODO  test4: 测试torch.normal(means= ,std= ,out=)    和test3一同测试
x0 = torch.normal(2*n_data, 1)  # 以[2, 2]为均值，1为标准差正态分布中随机取值，取10行
print(x0)

