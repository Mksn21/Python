import numpy as np 
import matplotlib.pyplot as plt
import math
x=np.zeros(20)
t=np.arange(-5,5,0.5)
a=0

for i in np.arange(0,20): 
  if t[i]>=-5.5+a and t[i]<=-4.5+a:
    x[i]=1
    a=a+1



      

print (x)
plt.plot(t,x)
plt.show()
print(a)
#print(len(t))
#print(t)
#print(x)