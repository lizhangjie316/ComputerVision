#处理丢失数据

import numpy as np
import pandas as pd

dates = pd.date_range('20130101',periods=6)

df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])

df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan

print(df.dropna(axis=0,how='any'))  #{any:只要数据中有nan，就会丢掉   all:所有数据全是nan才会丢掉}
            #通过 axis来控制是丢掉行还是列；axis=0 代表从上往下drop  删行
            # axis=1 代表从左向右删   删列

print(df.fillna(value=0))

print(df.isnull())

print(np.any(df.isnull()) == True)

#print(df)
