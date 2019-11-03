import numpy as np

a = np.random.random((2,4))

print(a)
print(a.sum())
print(np.sum(a,axis=1))
print(np.max(a,axis=0))
print(np.min(a))
