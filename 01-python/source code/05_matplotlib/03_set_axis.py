import numpy as np

import matplotlib.pyplot as plt

x = np.linspace(-3,3,50)

y1 = 2*x+1

y2 = x**2

plt.figure(num=3)


plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('I am x')
plt.ylabel('I am y')

new_ticks = np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3],
           [r'$really\ bad$',r'$bad\ \alpha$',r'$normal$',r'$good$',r'$really\ good$'])

#gca:  get current axis
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#ax.xaxis.set_ticks_position('bottom')
#ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))  #data   outward  axes
ax.spines['left'].set_position(('data',0))

l1,=plt.plot(x,y2,label='up')
l2,=plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--',label='down')

#增加图例
plt.legend(handles=[l1,l2],labels=['aaa','bbb'],loc='best')  #best   lower(upper) right(left)


plt.show()
