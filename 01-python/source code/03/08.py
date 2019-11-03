import numpy as np
import pandas as pd

a = list(range(4))
print(a)

s = pd.Series(1,index=list(range(1,5)),dtype='float32')

print(s)

arr = np.array([3]*4,dtype='int32')
print(arr)
