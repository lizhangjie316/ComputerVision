import numpy as np

a = np.arange(11,-1,-1).reshape(3,4)

print(a)

print(a.argmin())
print(np.argmin(a))
print(np.argmax(a))

print(np.median(a))  #中位数


print(np.mean(a,axis=1))  #求一行的平均值
print(np.mean(a,axis=0))  #求一列的平均值
print(a.mean())    #求所有的平均值
 
print(np.average(a,axis=1))

print(np.cumsum(a))  #累加:每一位皆为把原此位及原前边的所有 都加起来  共组成一行新的列表

print(np.diff(a))   #累差: 该位与横向下一位之差  故会将矩阵的列减少1

print(np.nonzero(a))  #(array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], dtype=int64))
                      # 左右两侧共同判定一个非0位

print(np.median(a))  #中位数


print(np.sort(a))  #将每一行进行单独排序

print(np.transpose(a))  #a的转置
print(a.T)

print(np.clip(a,5,9))  #a中所有小于5的置为5，所有大于9的置为9


