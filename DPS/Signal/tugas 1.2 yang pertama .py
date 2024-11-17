import numpy as np 
import matplotlib.pyplot as plt
import math
x=np.zeros(1000)
t=np.arange(-5,5,0.01)



for i in np.arange(0,1000):
 
    if t[i]>=-0.5 and t[i]<=0.5:
      x[i]=1 
    elif t[i]<1.5 and t[i]>2.5:
      x[i]=0
 
print (x)
plt.plot(t,x)
plt.show()
#print(len(t))
#print(t)
#print(x)