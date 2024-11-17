import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A1 = int(input('Amplitude 1 :'))
fo1 = int(input('freq 1 :'))
A2 = int(input('Amplitude 2 :'))
fo2 = int(input('freq 2 :'))
fs = int(input('Sampling Frequency :'))
N = int(input('N data :'))

#initial array
x = np.zeros(2000)
def signal(n):
    return A1*np.sin(2*fo1*np.pi*n/fs) + A2*np.sin(2*fo2*np.pi*n/fs)

for n in range(N):
    x[n] = signal(n)
    
#looping
n = np.arange(0,N,1,dtype=int)


#plot signal
print('Input Signal :')
plt.figure(figsize=((10,5)))
plt.plot(n/fs,x[n])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Time Domain")
plt.show()

#window rectanguler 
M=N 
w=np.zeros(2000)
xw=np.zeros(2000)


for n in range(M):
    w[n]=1

for n in range (M):
    xw[n]=x[n]*w[n]


n = np.arange(0,N,1,dtype=int)
#initial array
X_real = np.zeros(2000)
X_imaj = np.zeros(2000)
MagDFT = np.zeros(2000) 

#DFT
for k in range(N):
    for n in range(N):
        X_real[k] += xw[n]*np.cos(2*np.pi*k*n/N)
        X_imaj[k] -= xw[n]*np.sin(2*np.pi*k*n/N)
    
for k in range(N):
    MagDFT[k] = np.sqrt(np.square(X_real[k]) + np.square(X_imaj[k]))

n = np.arange(0,N,1,dtype=int)
k = np.arange(0,N,1,dtype=int)

fig, axs = plt.subplots(3, figsize=(13,9))
axs[0].plot(n/fs,w[n])
axs[0].set_title('Window',fontweight="bold", size=14)
axs[1].plot(n/fs,xw[n],'crimson')
axs[1].set_title('Xw(n)',fontweight="bold", size=14)
axs[2].stem(n/fs,MagDFT[n],'orange')
axs[2].set_title('DFT Xw(n)',fontweight="bold", size=14)
for ax in axs.flat:
    ax.set(xlabel='Waktu (s)',ylabel='Amplitudo (mV)')
    ax.grid()
axs[0].set(xlabel='Waktu(s)',ylabel='Widow')
fig.tight_layout(h_pad=3)