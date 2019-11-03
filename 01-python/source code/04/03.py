import numpy as np

import pandas as pd

df = pd.read_csv('student.csv')

print(df)
print(type(df))

df.to_pickle('student.pickle')


