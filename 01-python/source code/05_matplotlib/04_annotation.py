import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,50)
y = 2*x+1

plt.figure(num=1,figsize=(8,5))
plt.plot(x,y)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom') #设定x轴刻度位置 与 下 保持一致
ax.yaxis.set_ticks_position('left')

ax.spines['bottom'].set_position(('data',0)) #设定下脊梁位置 在y=0处
ax.spines['left'].set_position(('data',0))  #设定左脊梁（y轴）在x=0处


x0=1
y0=2*x0+1
plt.scatter(x0,y0,s=50,color='b')  #s:size    b:blue
plt.plot([x0,x0],[0,y0],'k--',lw=2.5)  #k:black  --:虚线  lw:linewidth

# method 1
################################
plt.annotate(r'$2x+1=%s$' % y0,xy=(x0,y0),xycoords='data',
             xytext=(+30,-30),textcoords='offset points',fontsize=16,
             arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))


# method 2
################################
plt.text(-3.7,3,r'$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',
         fontsize=16)  # fontsize={'size':16}

plt.show()
