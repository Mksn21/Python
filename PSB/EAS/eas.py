import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import signal
import plotly.graph_objects as go

df=pd.read_csv('Data.txt',sep='\s+',header=None)
df=pd.DataFrame(df)
df = df[:5000]
y = df[df.columns[0]]
fs = 75 
i = np.arange(len(y)) / 75



N = 5000 
n = np.arange(N)
k = n.reshape((N, 1))        
e = np.cos(2 * np.pi * k * n / N) - 1j * np.sin(2 * np.pi * k * n / N)

yy = np.dot(e,y)

yy = yy[:2500]
k = k[:2500]

stft_window = 500
f1, t1, ys = signal.stft(y, 75, nperseg=stft_window)



stft_window = 120
f, t, ys1 = signal.stft(y, 75, nperseg=stft_window)



freq = np.arange(1, 120)
w0 = 2 * np.pi * 0.849
scale = w0 * fs / (2 * freq * np.pi)
n = np.arange(N)
yc = signal.cwt(y, signal.morlet2, scale, dtype="complex128")




fig, axs = plt.subplots(5, figsize=(13,10))
axs[0].plot(i,y)
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude")
axs[0].set_title("Time Domain")
axs[1].stem(k*fs/5000,np.abs(yy), 'b', markerfmt=" ",
                              basefmt="-b")


axs[1].set_xlabel("Freq (Hz)")
axs[1].set_ylabel("Magnitude")
axs[1].set_title("Frequency Domain")

axs[2].pcolormesh((t1 * 75), f1, np.abs(ys), shading='gouraud')
axs[2].set_xlabel("Sequence(n)")
axs[2].set_ylabel("Frequency(hz)")
axs[2].set_title("STFT (500 dat/win)")

axs[3].pcolormesh((t * 75), f, np.abs(ys1), shading='gouraud')
axs[3].set_xlabel("Sequence(n)")
axs[3].set_ylabel("Frequency(hz)")
axs[3].set_title("STFT (120 dat/win)")


axs[4].pcolormesh(n, freq, np.abs(yc), shading='gouraud')
axs[4].set_xlabel("Sequence(n)")
axs[4].set_ylabel("Frequency(hz)")
axs[4].set_title("CWT")

plt.show()


