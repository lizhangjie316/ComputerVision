import numpy as np
import pandas as pd

dates = pd.date_range('20130101',periods=6)

df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])

print(df)

df.loc['20130102','B'] = 222

df.iloc[2,2] = 111

df.A[df.A<10] = 0

# df.F=np.nan  这种不能加新列
df['F'] = 0  # 这种可以加新列    np.nan
df['E'] = pd.Series([1,2,3,4,5,6],index=pd.date_range('20130101',periods=6)) #index应该对齐

print(df)
