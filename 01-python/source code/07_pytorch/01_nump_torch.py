# -*- encoding: utf-8 -*-
"""
@File    : 01_nump_torch.py
@Time    : 2019/7/25 10:00
@Author  : Keen
@Software: PyCharm
"""

import torch
import numpy as np

# np与numpy的互转     tensor即为torch的一种形式
np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print('numpy\n', np_data,
      'torch\n', torch_data,
      'tensor2array\n', tensor2array)

# abs sin mean
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)

print('\nnp.abs(tensor)', np.abs(tensor),
      '\ntensor.abs()', tensor.abs(),
      '\ntorch.abs(tensor)', torch.abs(tensor))

# 矩阵形式运算
data1 = [[1,2],[3,4]]

tensor = torch.FloatTensor(data1)
# 1. dot只支持一维，不再支持tensor
# 2. matmul与mm功能一致,numpy中没有mm，只有matmul
print('\nnumpy*numpy:',np.dot(data1,data1))
print('\nnumpy*numpy:',np.matmul(data1,data1))
print('\ntensor*tensor:',torch.matmul(tensor,tensor))
print('\ntensor*tensor:',torch.mm(tensor,tensor))

# 下边这项dot函数只可以进行1D的矩阵相乘，不能传入tensor(张量)为参数
# print('\ntensor*tensor:',torch.dot(tensor,tensor))

print(np.array(data1).dot(data1))
# print(tensor.dot(tensor)) 这种方式不再支持

print(tensor*tensor)  # 这样为对应项相乘，并非矩阵相乘
print(np.array(data1)**2)