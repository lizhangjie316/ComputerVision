# -*- encoding: utf-8 -*-
"""
@File    : imdb_classify.py
@Time    : 2019/8/7 9:35
@Author  : Keen
@Software: PyCharm
"""

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
i = 0
# print(train_data)
# print(len(train_data))
#
# print(len(test_data))
# print(max([max(sequence) for sequence in train_data]))
word_index = imdb.get_word_index()
# print(word_index)  # "oj's": 44864, 'gussied': 65111,
# print(word_index.items())  # dict_items([('fawn', 34701), ('tsukino', 52006)])  将字典中的每一项进行封装
# key:value => value:key
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# print("=xixi==========================================")
# print(reverse_word_index)
# print("=haha==========================================")

print(train_data)
print(len(train_data[0]))
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
print(decoded_review)

