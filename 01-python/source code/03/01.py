import numpy as np

a = np.array([[1,2],
              [1,0]])
b = np.arange(4).reshape(2,2)

c = a*b
c_dot = np.dot(a,b)
c_dot_2 = a.dot(b)

print(c)
print(c_dot)

print(c_dot_2)
