import numpy as np

import pandas as pd


df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])

df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])

df3 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])

print(df1)
print(df2)
print(df3)

#ignore_index 会将index进行重新重排
res = pd.concat([df1,df2,df3],axis=0,ignore_index=True)

print(res)

res = pd.concat([df1,df2,df3],axis=1,ignore_index=True)  
print(res)


#join,['inner','outer' 默认为outer]   以下两个df有所不同
df4 = pd.DataFrame(np.ones((3,4))*0,index=[1,2,3],columns=['a','b','c','d'])

df5 = pd.DataFrame(np.ones((3,4))*1,index=[2,3,4],columns=['b','c','d','e'])

print(df4)
print(df5)
#默认axis为0
res = pd.concat([df4,df5],axis=0,join='inner',ignore_index=True)
print(res)

#join_axes
#按照df4的索引进行合并，df5中有的，但df4中没有的不输出来
res = pd.concat([df4,df5],axis=1,join_axes=[df4.index])  
print(res)

#append  默认往下加数据 即axis=0
res = df1.append([df2,df3],ignore_index=True)
print(res)

#添加一条数据
s1 = pd.Series([1,2,3,4],index=['a','b','c','d'])
res = df1.append(s1,ignore_index=True)
print(res)















