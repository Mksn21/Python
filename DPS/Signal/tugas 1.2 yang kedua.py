import numpy  as np
import matplotlib.pyplot as plt
import math


x=np.zeros(1000)
t=np.arange(-5,5,0.01)
b=np.zeros(1000)

for i in np.arange(0,1000): 
   for j in range (-4,5,2):
      if t[i]>=j-0.5 and t[i]<=j+0.5:
        x[i]=1
   



plt.plot(t,x)
plt.show()
#print(x)