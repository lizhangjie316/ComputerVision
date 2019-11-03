# -*- encoding: utf-8 -*-
"""
@File    : 05_minist_convnet.py
@Time    : 2019/8/20 11:10
@Author  : Keen
@Software: PyCharm
"""

from keras import layers
from keras import models
import numpy as np

# 构建卷积神经网络
model = models.Sequential()
# filter  activation  input_shape(h,w,c)
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

# 观察CNN的结构
# model.summary()

# 构建全连接网络
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# model.summary()

from keras.datasets import mnist
from keras.utils import to_categorical

# 导入数据
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32')/255

# 对标签进行分类编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 进行训练
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# 进行测试
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)