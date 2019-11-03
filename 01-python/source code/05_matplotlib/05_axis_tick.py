import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-3,3,50)
y=0.1*x

plt.plot(x,y,linewidth=10,zorder=0)
plt.ylim(-2,2)

#1 处理坐标线
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

#print(ax.get_xticklabels())  #标签对象的列表
#print(ax.get_xticklabels()+ax.get_yticklabels())  #标签对象的列表

#处理被覆盖的标签
for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_fontsize(12) #{'facecolor':'white','edgecolor':'None','alpha':0.7}
    label.set_bbox(dict(facecolor='white',edgecolor='None',alpha=0.7))
    label.set_zorder(1)


plt.show()
