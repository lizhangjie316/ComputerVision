import numpy as np


array = np.array([[1,2,3],
                 [4,5,6]],dtype=np.float)

a1 = np.zeros((3,4))

a2 = np.ones((3,4))

a3 = np.arange(10,20,2)

a4 = np.arange(12).reshape((3,4))

a5 = np.linspace(1,10,4).reshape(2,2)
print(a5)


print(a1)

print(a2)

print(array.dtype)

print(array)

print('number of dim:',array.ndim)
print('shape:',array.shape)
print('size:',array.size)

a2 = np.empty((3,3))

print(a2)
