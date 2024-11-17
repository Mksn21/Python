import numpy as np 
import matplotlib.pyplot as plt
import math

def sinyal_Xn(n):
  x=0
  #selain yang didefinisikan di bawah ini, nilai x=0
  if n>=-2 and n<=0:
    x=2+n
  if n==1:
    x=0.5
  if n==2:
    x=1
  if n==3:
    x=2
  return x

# cara pemakaiannya:
n=np.arange(-6,7,1)
x=np.zeros(len(n))
for i in np.arange(0,len(n)):
  x[i]=sinyal_Xn(n[i])

# kita akan mendapatkan plot yang sama
plt.stem(n,x,use_line_collection=True)
plt.show()