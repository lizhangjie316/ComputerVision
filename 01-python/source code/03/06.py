import numpy as np

A = np.arange(12).reshape((3,4))

print(A)

print(np.split(A,1,axis=0))  #从上至下进行分割，分为1块

print(np.split(A,2,axis=1))  #从左至右进行分割，分为2块

print(np.split(A,3,axis=0))  #从上至下分，分为3块



print(np.vsplit(A,3))

print(np.hsplit(A,2))

print(np.array_split(A,3,axis=1))   #唯一可以不等分割的
