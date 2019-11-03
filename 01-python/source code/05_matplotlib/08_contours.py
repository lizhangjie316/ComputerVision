import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    #height
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)


n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)

X,Y=np.meshgrid(x,y)
#print(X)
#print('=======================')
#print(Y)

#填充等高面
plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plt.cm.hot)

#作等高线
C=plt.contour(X,Y,f(X,Y),8,colors='black',linewidth=.5)
#添加标签
plt.clabel(C,inline=True,fontsize=10)


plt.show()
