import numpy as np

A = np.array([1,1,1]) 
B = np.array([2,2,2])

print(A)
print(B)

C = np.vstack((A,B,B,A))
D = np.hstack((A,B))

print(C)
print(D)

A = np.array([1,1,1])[:,np.newaxis]  
B = np.array([2,2,2])[:,np.newaxis]

print(np.concatenate((A,B,B),axis=1))
