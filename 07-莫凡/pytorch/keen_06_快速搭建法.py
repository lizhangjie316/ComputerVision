# -*- encoding: utf-8 -*-
"""
@File    : keen_06_快速搭建法.py
@Time    : 2019/11/4 8:41
@Author  : Keen
@Software: PyCharm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net1, self).__init__()
		self.hidden = nn.Linear(n_feature, n_hidden)
		self.predict = nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x = F.relu(self.hidden(x))
		x = self.predict(x)
		return x


net1 = Net1(1, 10, 1)
print(net1)
# Net1(
#   (hidden): Linear(in_features=1, out_features=10, bias=True)
#   (predict): Linear(in_features=10, out_features=1, bias=True)
# )

net2 = nn.Sequential(
	nn.Linear(1, 10),
	nn.ReLU(),
	nn.Linear(10, 1)
)
print(net2)
# Sequential(
#   (0): Linear(in_features=1, out_features=10, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=10, out_features=1, bias=True)
# )

