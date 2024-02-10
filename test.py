import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
tp=2*np.pi

fig,ax=plt.subplots()
x=np.arange(0.0,tp,0.001)
s=np.sin(x)
pl=plt.plot(x,s)

ax=plt.axis([0,tp,-1,1])
redDot,=plt.plot([0],[np.sin(0)],'ro')

def animate(i):
  redDot.set_data(i,np.sin(i))
  return redDot,

myanimation = animation.FuncAnimation(fig,animate,frames=np.arange(0.0,tp,0.1),interval=10,repeat=True)
plt.show()