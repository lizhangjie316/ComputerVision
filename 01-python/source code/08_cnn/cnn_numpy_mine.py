# -*- encoding: utf-8 -*-
"""
@File    : cnn_numpy_qi.py
@Time    : 2019/7/30 20:28
@Author  : Keen
@Software: PyCharm
"""
# import numpy as np
# import math
# import tensorflow
# from tensorflow.python.framework import dtypes
#
# def im2col(image, ksize, stride):
#     # image is a 4d tensor([batchsize, width, height, channel])
#     image_col = []
#     for i in range(0, image.shape[1] - ksize + 1, stride):
#         for j in range(0, image.shape[2] - ksize +1, stride):
#             col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
#             image_col.append(col)
#     image_col = np.array(image_col)
#     return image_col
#
# class Conv2D(object):
#     def __init_(self, shape, output, output_channels, ksize=3, stride=1, method='VALID'):
#         self.input_shape = shape
#         self.output_channels = output_channels
#         self.input_channels = shape[-1]
#         self.batchsize = shape[0]
#         self.stride = stride
#         self.ksize = ksize
#         self.method = method
#         weights_scale = math.sqrt(ksize*ksize*self.input_channels/2)
#         # /weights_scale "msra" init
#         self.weights = np.random.standard_normal((ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
#         self.bias = np.random.standard_normal(self.output_channels) / weights_scale
#
#         if method == 'VALID':
#             self.eta = np.zeros((shape[0], (shape[1] - ksize) / self.stride + 1, (shape[1] - ksize) / self.stride + 1, self.output_channels))
#         if method == 'SAME':
#             self.eta = np.zeros((shape[0], shape[1]/self.stride, shape[2]/self.stride, self.output_channels))
#
#         self.w_gradient = np.zeros(self.weights.shape)
#         self.b_gradient = np.zeros(self.bias.shape)
#         self.output_shape = self.eta.shape
#
#     def forward(self, x):
#         col_weights = self.weights.reshape([-1, self.output_channels])
#         if self.method == 'SAME':
#             x = np.pad(x, ((0,0), (self.ksize/2, self.ksize/2), (self.ksize/2, self.ksize/2), (0,0)), 'constant', constant_values=0)
#
#         self.col_image = []
#         conv_out = np.zeros(self.eta.shape)
#         for i in range(self.batchsize):
#             img_i = x[i][np.newaxis, :]
#             self.col_image_i = im2col(img_i, self.ksize, self.ksize, self.stride)
#             conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
#             self.col_image.append(self.col_image_i)
#         self.col_image = np.array(self.col_image)
#         return conv_out
#
# if __name__ == "__main__":
#     conv1 = Conv2D([4, 28, 28, 1], 12, 5, 1)


# !---test
# ksize = 3
# input_channels = 3
# output_channels = 2
# weights = np.random.standard_normal((2, 2, 3, 2))
# print(weights.shape)
# print(weights)
# col_weights = weights.reshape([-1, output_channels])
# print(col_weights.shape)
# print(col_weights)

# z = np.random.standard_normal((1,3,4,3))
# col = z[:, 0:0 + 2, 0:0 + 2, :]
# col1 = col.reshape([-1])
# print(z.shape)
# print(z)
# print(col.shape)
# print(col)
# print(col1.shape)
# print(col1)
# !---test over


import numpy as np
stride = 2
nf = 2 # number of filters
sf = 3 # filter size
input_h = 7
input_w = 7
input_c = 3
padding = 0
batchsize = 1
output_h = int((input_h - sf + padding) / stride) + 1
output_w = int((input_w - sf + padding) / stride) + 1
output_c = nf
np_o = np.zeros((batchsize, output_h, output_w, output_c))
np_o_fast = np_o

# setting up initial values
x = np.zeros((input_h, input_w, input_c)) # height width channel
x[:, :, 0] = np.mat("0 0 0 0 0 0 0;0 0 1 0 1 0 0;0 2 1 0 1 2 0;0 0 2 0 0 1 0;0 2 0 1 0 0 0;0 0 0 1 2 2 0;0 0 0 0 0 0 0").A
x[:, :, 1] = np.mat("0 0 0 0 0 0 0;0 1 0 0 1 1 0;0 0 2 2 1 1 0;0 2 1 2 1 0 0;0 2 1 1 2 2 0;0 1 2 0 2 2 0;0 0 0 0 0 0 0").A
x[:, :, 2] = np.mat("0 0 0 0 0 0 0;0 2 1 1 1 1 0;0 2 2 1 2 1 0;0 1 1 0 2 2 0;0 2 1 2 2 0 0;0 1 2 2 0 0 0;0 0 0 0 0 0 0").A

x = np.reshape(x, (batchsize, input_h, input_w, input_c)) # batch height width channel
#print("x:\n", x)

w = np.zeros((3, 3, 3, 2)) # height width channel num
w[:, :, 0, 0] = np.mat("0 0 1;-1 1 1;0 1 0").A
w[:, :, 1, 0] = np.mat("1 1 1;0 1 1;0 1 0").A
w[:, :, 2, 0] = np.mat("-1 0 0;-1 1 1;0 -1 0").A

w[:, :, 0, 1] = np.mat("0 0 0;1 1 -1;-1 1 1").A
w[:, :, 1, 1] = np.mat("0 1 -1;1 1 -1;-1 1 -1").A
w[:, :, 2, 1] = np.mat("1 1 0;-1 -1 0;0 -1 1").A

bias = [1, 0]
#print("w:\n", w)

# CNN in numpy
print("---Convolution in numpy---")
for z in range(nf):
    for _h in range(output_h):
        for _w in range(output_w):
            input_temp = x[0, _h*stride:_h*stride+sf, _w*stride:_w*stride+sf, :] #shape (3, 3, 3)
            np_o[0, _h, _w, z] = np.sum(input_temp * w[:, :, :, z]) + bias[z] #w[:,:,:,z].shape: (3, 3, 3)

print("np_o0:\n", np_o[0, :, :, 0])
print("np_o1:\n", np_o[0, :, :, 1])

# CNN in numpy fast
print("---Convolution in numpy fast---")

def im2col(image, ksize, stride):
    # image: [batch, height, width, channel]
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i+ksize, j:j+ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col

col_weights = w.reshape([-1, nf])
# print("col_weights:\n", col_weights)
col_image = []
for i in range(1):
    img_i = x[i][np.newaxis, :]
    col_image_i = im2col(img_i, 3, 2)
    # print("col_image_i:\n", col_image_i)
    np_o_fast[i] = np.reshape(np.dot(col_image_i, col_weights) + bias, (3, 3, 2))
    #col_image.append(col_image_i)
#col_image = np.array(col_image)

print("np_o_fast0:\n", np_o_fast[0, :, :, 0])
print("np_o_fast1:\n", np_o_fast[0, :, :, 1])


