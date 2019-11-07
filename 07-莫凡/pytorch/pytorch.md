<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
[TOC]
----
## 1. 激励函数分类
```
        x = torch.linspace(-5, 5, 200)
        y_sigmoid = torch.sigmoid(x)
        y_relu = F.relu(x)
        y_tanh = torch.tanh(x)
        y_softplus = F.softplus(x)
```

![image](./../../img_fold/激励函数图.png "激活函数图")

## 2. 构建分类网络
### 1. 造数据
```
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 1:从上往下加维度 shape: [100, 1]    0:从左往右加维度 shape: [1,100]
y = x.pow(2) + 0.2 * torch.rand(x.size()) # y = x^2 + 高斯白噪声  从0-1中取x.size()个值
```
### 2. 建立神经网路
```
class Net(nn.Module):

	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.hidden = nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
		self.predict = nn.Linear(n_hidden, n_output)  # 输出层线性输出

	# 重写了module中的forward
	def forward(self, x):
		x = F.relu(self.hidden(x))
		x = self.predict(x)
		return x
```
### 3. 实例化网络
```
net = Net(n_feature=1, n_hidden=10, n_output=1)
```
### 4. 训练网络
#### 4.1 准备优化器及损失函数
```
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入net的所有参数，学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值之间的误差计算公式(MSE)   均方差损失
```
### 5. 可视化训练过程
```
plt.ion()  # 交互式画图开启

for t in range(150):  # 迭代对数据进行训练

	prediction = net(x)  # input x and predict based on x

	loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)   计算两者间的误差

	optimizer.zero_grad()  # clear gradients for next train  清空上一步的残余更新参数
	loss.backward()  # backpropagation, compute gradients  误差反向传播，计算参数更新值
	optimizer.step()  # apply gradients  将参数更新值施加到net的paramters上，即对参数进行优化

	if t % 5 == 0:
		# plot and show learning process
		plt.cla()
		plt.scatter(x, y)
		plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
		plt.text(0.5, 0, 'Loss=%.4f' % loss, fontdict={'size': 20, 'color': 'red'})
		plt.pause(0.1)

plt.ioff() # 交互式画图关闭
plt.show()
```

## 10.卷积神经网络 
### 10.1 卷积
##  卷积性质
1. 卷积核的个数决定卷积生成特征图的通道数
2. 输入特征图的通道数决定单个卷积核的深度
3. 通常一个kernel包含多层filter
4. 卷积操作后的特征图的尺度满足： ![image](./../../img_fold/求卷积后宽度.png "卷积后宽度")
5. 当stride=1时，想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2

## 卷积原则

- 卷积过程不压缩长宽，只改变channel数，尽量保留更多的信息
- polling阶段压缩长宽，不改变channel数





