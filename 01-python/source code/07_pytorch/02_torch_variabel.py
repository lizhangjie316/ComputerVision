# -*- encoding: utf-8 -*-
"""
@File    : 02_torch_variabel.py
@Time    : 2019/7/25 11:23
@Author  : Keen
@Software: PyCharm
"""

import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])

# 设置requires_grad=True则会追踪对于该张量的操作
variable = Variable(tensor, requires_grad=True)

print(tensor)
print(variable)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable) # x^2

print(t_out)
print(v_out)

v_out.backward() # 平均值的反向传递; 与variable相关的值发生改变，则
# v_out(var) = 1/4*sum(var*var)
# variable.grad = d(v_out)/d(var) = 1/4*2*variable = variable/2

print(variable) #仍旧是原来的形式

print(variable.grad)
print(variable.data)
print(variable.data.numpy())








