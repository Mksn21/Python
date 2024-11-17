import numpy as np
import matplotlib.pyplot as plt

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

#initial array
X_real = np.zeros(2000)
X_imaj = np.zeros(2000)
MagDFT = np.zeros(2000) 

#DFT
for k in range(N):
    for n in range(N):
        X_real[k] += x[n]*np.cos(2*np.pi*k*n/N)
        X_imaj[k] -= x[n]*np.sin(2*np.pi*k*n/N)
    
for k in range(N):
    MagDFT[k] = np.sqrt(np.square(X_real[k]) + np.square(X_imaj[k])) 

#looping
n = np.arange(0,N,1,dtype=int)
k = np.arange(0,N,1,dtype=int)     
    
#plotting sinyal DFT
print('DFT :')
plt.figure(figsize=((10,5)))
plt.stem(k*fs/N, MagDFT[k])
plt.xlabel("Freq (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Domain")
plt.show()

y =np.zeros(2000)



#IDFT 
for n in range(N):
    for k in range(N):
        y[n]+=(X_real[k]-(X_imaj[k]))*((np.cos(2*np.pi*k*n/N))+(np.sin(2*np.pi*k*n/N)))/N

#looping
n = np.arange(0,N,1,dtype=int)
k = np.arange(0,N,1,dtype=int)     

print('IDFT:')
plt.figure(figsize=((10,5)))
plt.plot(n/fs,y[n])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Time Domain")
plt.show()

