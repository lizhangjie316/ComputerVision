import numpy as np
import pandas as pd

dates = pd.date_range('20130101',periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])

print(df)


print(df['A'])  #取得列上面的
print(df.A)
  
print(df[0:3])   #取得行上面的
print(df['20130101':'20130104'])   #20130104的数据也会被打印出来

#select by label
print(df.loc['20130102'])

print(df.loc[:,['A','B']])  #保留所有的行，并打出A B两列的数据

print(df.loc['20130102',['A','B']])


#select by position :iloc (index loc)
print(df.iloc[3])

print(df.iloc[3,0])

print(df.iloc[3:5,1:3])

print(df.iloc[[1,3,5],:])

#mixed selection:ix
print(df.ix[3,['A','B','C']])  #这种做法已经落伍

#Boolean indexing 用判断来进行筛选
print(df[df.A>8])  #选出A大于8的所有行

