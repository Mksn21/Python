
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



#memasukkan data
df = pd.read_excel('SuddecCardiacDeathv3.xlsx', header=[0,1])
t = df[df.columns[0]]
s1 = df[df.columns[1]]
s2 = df[df.columns[2]]
fs = 1/0.004
N = len(df.index)


fig, axs = plt.subplots(2, figsize=(13,9))
axs[0].plot(t*0.004,s1)
axs[0].set_title('Sinyal ECG 1',fontweight="bold", size=14)
axs[1].plot(t*0.004,s2,'crimson')
axs[1].set_title('Sinyal ECG 2',fontweight="bold", size=14)
for ax in axs.flat:
    ax.set(xlabel='Waktu (s)',ylabel='Amplitudo (mV)', xlim=(0,10), xticks=(np.arange(0,11)))
    ax.grid()
fig.tight_layout(h_pad=3)

X_real1 = np.zeros(N)
X_imaj1 = np.zeros(N)
MagDFT1 = np.zeros(N)
X_real2 = np.zeros(N)
X_imaj2 = np.zeros(N)
MagDFT2 = np.zeros(N)

for k in range(N):
    for n in range(N):
        X_real1[k] += s1[n]*np.cos(2*np.pi*k*n/N)
        X_imaj1[k] -= s1[n]*np.sin(2*np.pi*k*n/N)
    
for k in range(N):
    MagDFT1[k] = np.sqrt(np.square(X_real1[k]) + np.square(X_imaj1[k])) 
    
for k in range(N):
    for n in range(N):
        X_real2[k] += s2[n]*np.cos(2*np.pi*k*n/N)
        X_imaj2[k] -= s2[n]*np.sin(2*np.pi*k*n/N)
    
for k in range(N):
    MagDFT2[k] = np.sqrt(np.square(X_real2[k]) + np.square(X_imaj2[k])) 
    
n = np.arange(0,N,1,dtype=int)
k = np.arange(0,N,1,dtype=int)

fig, axs = plt.subplots(2, figsize=(13,9))
axs[0].stem(k*fs/N,MagDFT1[n])
axs[0].set_title('Sinyal DFT ECG 1',fontweight="bold", size=14)
axs[1].stem(k*fs/N,MagDFT2[n],'crimson')
axs[1].set_title('Sinyal DFT ECG 2',fontweight="bold", size=14)
for ax in axs.flat:
    ax.set(xlabel='Freq (Hz)',ylabel='Amplitudo (mV)', xlim=(0,20), xticks=(np.arange(0,20)))
    ax.grid()
fig.tight_layout(h_pad=3)

y1 =np.zeros(N)
y2 =np.zeros(N)

for n in range(N):
    for k in range(N):
        y1[n]+=(X_real1[k]-(X_imaj1[k]))*((np.cos(2*np.pi*k*n/N))+(np.sin(2*np.pi*k*n/N)))/N
        y2[n]+=(X_real2[k]-(X_imaj2[k]))*((np.cos(2*np.pi*k*n/N))+(np.sin(2*np.pi*k*n/N)))/N

#looping
n = np.arange(0,N,1,dtype=int)
k = np.arange(0,N,1,dtype=int)  

fig, axs = plt.subplots(2, figsize=(13,9))
axs[0].plot(t*0.004,y1[n])
axs[0].set_title('Sinyal ECG 1',fontweight="bold", size=14)
axs[1].plot(t*0.004,y2[n],'crimson')
axs[1].set_title('Sinyal ECG 2',fontweight="bold", size=14)
for ax in axs.flat:
    ax.set(xlabel='Waktu (s)',ylabel='Amplitudo (mV)', xlim=(0,10), xticks=(np.arange(0,11)))
    ax.grid()
fig.tight_layout(h_pad=3)