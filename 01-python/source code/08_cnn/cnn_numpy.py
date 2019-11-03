# -*- encoding: utf-8 -*-
"""
@File    : cnn_numpy.py
@Time    : 2019/7/30 13:53
@Author  : Keen
@Software: PyCharm
"""

import numpy as np

stride = 2
nf = 2  # number of filter
fs = 3  # filter size
input_h = 7
input_w = 7
input_c = 3
padding = 0
batchsize = 1  # 一次训练所选取的样本数

output_h = int((input_h - fs + padding)/stride) + 1  # 3
output_w = int((input_w - fs + padding)/stride) + 1  # 3
output_c = nf
np_o = np.zeros((batchsize, output_h, output_w, output_c))
np_o_fast = np_o

# setting up initial values
x = np.zeros((input_h, input_w, input_c))  # height width channel
x[:, :, 0] = np.mat("0 0 0 0 0 0 0;0 0 1 0 1 0 0;0 2 1 0 1 2 0;0 0 2 0 0 1 0;0 2 0 1 0 0 0; 0 0 0 1 2 2 0;0 0 0 0 0 0 0").A  # trans to array
x[:, :, 1] = np.mat("0 0 0 0 0 0 0;0 1 0 0 1 1 0;0 0 2 2 1 1 0;0 2 1 2 1 0 0;0 2 1 1 2 2 0; 0 1 2 0 2 2 0;0 0 0 0 0 0 0").A
x[:, :, 2] = np.mat("0 0 0 0 0 0 0;0 2 1 1 1 1 0;0 2 2 1 2 1 0;0 1 1 0 2 2 0;0 2 1 2 2 0 0; 0 1 2 2 0 0 0;0 0 0 0 0 0 0").A

# print('x:\n', x[:, :, 2])

x = np.reshape(x, (batchsize, input_h, input_w, input_c))#(1,7,7,3)

w = np.zeros((3, 3, 3, 2))  # height width channel num
w[:, :, 0, 0] = np.mat("0, 0, 1;-1 1 1;0 1 0").A
w[:, :, 1, 0] = np.mat("1 1 1;0 1 1;0 1 0").A
w[:, :, 2, 0] = np.mat("-1 0 0;-1 1 1;0 -1 0").A

w[:, :, 0, 1] = np.mat("0 0 0;1 1 -1;-1 1 1").A
w[:, :, 1, 1] = np.mat("0 1 -1;1 1 -1;-1 1 -1").A
w[:, :, 2, 1] = np.mat("1 1 0; -1 -1 0; 0 -1 1").A


bias = [1,0]

# CNN in numpy
print("---Convolution in numpy---")

for z in range(nf):
    for _h in range(output_h):
        for _w in range(output_w):

            input_temp = x[0, _h*stride: _h*stride + fs, _w*stride : _w*stride+fs, :]  # shape(3, 3, 3)  选取到具体的某一块
            #print(input_temp[:,:,0])
            np_o[0, _h, _w, z] = np.sum(input_temp * w[:, :, :, z]) + bias[z]  # w[:,:,:,z].shape: (3,3,3)

print("np_o0:\n",np_o[0, :, :, 0])
print("np_o1:\n",np_o[0, :, :, 1])

print("---fast convolution---")
def im2col(image, ksize, stride):  # trans the image to martrix
    # image or x: [batch, height, width, channel]
    image_col = []
    # print("image.shape[0]:",image.shape[0])  # 1
    # print("image.shape[1]:",image.shape[1])  # 7 行
    # print("image.shape[2]:",image.shape[2])  # 7 列
    for i in range(0, image.shape[1] - ksize + 1, stride):  # 新构成的行
        for j in range(0, image.shape[2] - ksize + 1, stride): # 新构成的列
            col = image[:, i:i+ksize, j:j+ksize, :].reshape([-1])  # 3通道的所有的(0,0)号位、(0,1)号...依次排开
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col

col_weights = w.reshape([-1, nf])  # 将w矩阵变成列矩阵，且一行为nf个
# print("col_weights:\n", col_weights)
col_image = []
for i in range(1):  # 依次取出所有样本；     1 在此处代表一次取0个样本；
    img_i = x[i][np.newaxis, :]   # 在x[i]的横向上加了一个维度： 在外侧再加一对[ ]
    # print("x[i]:\n", x[i])
    print("===========================================")
    # print("img_i:\n", img_i)
    col_image_i = im2col(img_i, 3, 2)  # 将获取的当前样本进行转换成特定的数组，ksize==3, stride==2
    print("col_image_i:\n", col_image_i)
    print("col_weights.shape:", col_weights.shape)
    print("col_image_i.shape:", col_image_i.shape)
    np_o_fast[i] = np.reshape(np.dot(col_image_i, col_weights) + bias, (3, 3, 2)) #reshape(x,shape)将x数据转换成(3,3,2) 3行3列2通道
    #col_image.append(col_image_i)
#col_image = np.array(col_image)



print("np_o_fast0:\n", np_o_fast[0, :, :, 0])
print("np_o_fast1:\n", np_o_fast[0, :, :, 1])
