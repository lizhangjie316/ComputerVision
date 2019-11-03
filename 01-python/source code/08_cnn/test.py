# -*- encoding: utf-8 -*-
"""
@File    : test.py
@Time    : 2019/7/31 10:31
@Author  : Keen
@Software: PyCharm
"""

import numpy as np

arr1 = np.array([1,2,3,4,5,6,7,8,9,10])
# arr1 = np.array([[1,2,3,4,5,6,7,8,9,10],
#               [1,2,3,4,5,6,7,8,9,10]])
print("arr1:\n", arr1)
print("arr1.shape:", arr1.shape)
arr2 = arr1.reshape([-1])  # 去掉所有的维度，排成一排
#  [ 1  2  3  4  5  6  7  8  9 10]
# arr2.shape: (10,)

# 横向加一维
# arr2 = arr2[np.newaxis,:]
# [[ 1  2  3  4  5  6  7  8  9 10]]
# arr2.shape: (1, 10)

# 纵向加一维
# arr2 = arr2[:, np.newaxis]
# arr2:
#  [[ 1]
#  [ 2]
#  [ 3]
#  [ 4]
#  [ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]]
# arr2.shape: (10, 1)
print("arr2:\n", arr2)
print("arr2.shape:", arr2.shape)

arr3 = arr1.reshape([-1,2])  # 排成纵列，每行2个
# arr3:
#  [[ 1  2]
#  [ 3  4]
#  [ 5  6]
#  [ 7  8]
#  [ 9 10]]
# arr3.shape: (5, 2)
print("arr3:\n", arr3)
print("arr3.shape:", arr3.shape)

arr4 = []
arr5 = [1,2,3,4]
arr4 = arr4.append(arr5)

# print("arr4.dtype:",arr4.dtype)  nonetype
print("arr4:", arr4)  # None
