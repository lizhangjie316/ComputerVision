import numpy as np

A = np.arange(3,15).reshape((3,4))

print(A)

print(A[2])

print(A[2][1])

print(A[2,1])

print(A[2,:])

print(A[:,1])

print(A[1,1:2])

for row in A:   #for循环默认迭代的是每一行
    print(row)

for column in A.T:  #这种方式迭代每一列
    print(column)

print(A.flat)
print(A.flatten())

for item in A.flat:
    print(item,end=" ")

