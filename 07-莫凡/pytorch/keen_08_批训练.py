# -*- encoding: utf-8 -*-
"""
@File    : keen_08_批训练.py
@Time    : 2019/11/5 19:34
@Author  : Keen
@Software: PyCharm
"""
import torch
import torch.utils.data as Data

torch.manual_seed(1)

"""
本期学习：
	1. DataLoader： 一个类，用来包装要处理的数据，从而更有效的迭代数据
	2. 批训练：
	
	

"""

# 5个数据共同进行一次处理
# BATCH_SIZE = 5
BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# 1. 先将初试数据进行包装
torch_dataset = Data.TensorDataset(x, y)
# 2. 将包装后的数据封装到DataLoader类中
loader = Data.DataLoader(
	dataset=torch_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True,  # 是否打乱数据
	num_workers=2  # 多线程来读取数据
)


def show_batch():
	for epoch in range(3): # 3. 训练数据3轮
		# 4. 每步导出5个数据
		for step, (batch_x, batch_y) in enumerate(loader):  # enumerate(loader) 返回枚举对象，以索引、及其对应位置数据 的形式
			# train data
			print("Epoch: ", epoch, "| Step: ", step, "batch x: ",
			      batch_x.numpy(), "| batch y: ", batch_y.numpy())


if __name__ == "__main__":
	show_batch()


