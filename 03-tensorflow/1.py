# -*- encoding: utf-8 -*-
"""
@File    : 1.py
@Time    : 2019/8/5 19:29
@Author  : Keen
@Software: PyCharm
"""

import numpy as np
import tensorflow as tf

# arr = np.arange(16).reshape((2, 8))
# #
# # print(arr)

＃在我的Youtube和优酷频道上查看更多python学习教程！

＃的Youtube视频教程：https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
＃优酷视频教程：http：//i.youku.com/pythontutorial

'''
请注意，此代码仅适用于python 3+。如果您使用的是python 2+，请相应地修改代码。
'''
#来自 __future__
import print_function
#导入张量流为 tf


def  add_layer（inputs，in_size，out_size，activation_function = None）：
    权重= tf.Variable（tf.random_normal（[in_size，out_size]））
    偏置= tf.Variable（tf.zeros（[ 1，out_size]）+  0.1）
    Wx_plus_b = tf.matmul（输入，权重）+偏差
    如果 activation_function 为 None：
        outputs = Wx_plus_b
    否则：
        outputs = activation_function（Wx_plus_b）
    返回输出