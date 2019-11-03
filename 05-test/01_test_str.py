# -*- encoding: utf-8 -*-
"""
@File    : 01_test_str.py
@Time    : 2019/8/23 16:22
@Author  : Keen
@Software: PyCharm
"""

import os

total_name = os.listdir('../')

print(total_name)
list = range(len(total_name))     # [0,1,...,len(total_name)-1]

for i in list:
	name = total_name[i][:-4]
	print(name)