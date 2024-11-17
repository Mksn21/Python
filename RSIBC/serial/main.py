from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
import pandas as pd
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import numpy as np
from scipy.io.wavfile import read
from scipy.fft import fft, fftfreq
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
import sklearn.metrics as met
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import serial
import threading
import time

class widgetss (QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("a.ui",self)
        self.is_collecting = False
        self.audio = []
        self.fs = 4000
        self.addToolBar(NavigationToolbar(self.widget.canvas, self))
        self.pushButton.clicked.connect(self.inputS)
        self.pushButton_2.clicked.connect(self.DataS)
        self.pushButton_3.clicked.connect(self.pre)
        self.horizontalSlider.valueChanged.connect(self.update)
        self.pushButton_4.clicked.connect(self.thr)
        self.pushButton_6.clicked.connect(self.env)
        self.pushButton_5.clicked.connect(self.hsl)
        self.pushButton_7.clicked.connect(self.stop_collecting)

    def inputS(self):
        if not self.is_collecting:
            try:
                #the arduino is connected to the COM6 port with baudrate=9600
                self.serial_port = serial.Serial(port="COM10", baudrate=115200, timeout=1)
                self.is_collecting = True
                #create array to collect the data
                self.audio = []
                self.data_collection_thread = threading.Thread(target=self.collect_data)
                self.data_collection_thread.start()
            except Exception as e:
                print(f"Error opening COM port: {e}")

    def stop_collecting(self):
        if self.is_collecting:
            self.is_collecting = False
        if self.serial_port:
            self.serial_port.close()
        self.a = np.arange(len(self.audio))
        self.widget.canvas.axes.clear()
        self.widget.canvas.axes.set_title('Signal')
        self.widget.canvas.axes.set_xlabel('time')
        self.widget.canvas.axes.set_ylabel('Amplitude')
        self.widget.canvas.axes.plot(self.a/self.fs,self.audio,label="Input")
        self.widget.canvas.axes.legend()
        self.widget.canvas.axes.grid()
        self.widget.canvas.draw()

    def collect_data(self):
        while self.is_collecting:
            try:
                line = self.serial_port.readline().decode('ascii')
                if line:
                    self.audio.append(float(line))
                self.plot_spectrum()
                time.sleep(1/10)
            except Exception as e:
                print(f"Error reading from COM port: {e}")

    def plot_spectrum(self):
        if self.is_collecting:
            self.a = np.arange(len(self.audio))
            self.widget.canvas.axes.clear()
            self.widget.canvas.axes.set_title('Signal')
            self.widget.canvas.axes.set_xlabel('time')
            self.widget.canvas.axes.set_ylabel('Amplitude')
            self.widget.canvas.axes.plot(self.a/self.fs,self.audio,label="Input")
            self.widget.canvas.axes.legend()
            self.widget.canvas.axes.grid()
            self.widget.canvas.draw()

    def DataS(self):
        i = int(self.lineEdit.text())
        j = int(self.lineEdit_2.text())
        self.audio1 = self.audio[i:j] / np.max(self.audio)
        self.fs = 4000
        self.a = np.arange(len(self.audio1))
        self.widget.canvas.axes.clear()
        self.widget.canvas.axes.set_title('Signal')
        self.widget.canvas.axes.set_xlabel('time')
        self.widget.canvas.axes.set_ylabel('Amplitude')
        self.widget.canvas.axes.plot(self.a/self.fs,self.audio1,label="Input")
        self.widget.canvas.axes.legend()
        self.widget.canvas.axes.grid()
        self.widget.canvas.draw()    
 
    def pre(self):
        self.N = len(self.audio1)
        fcd = np.pi / self.N 
        Ndw = 2 
        def lpffir (omc,i):
            if i == 0 :
                y = omc/np.pi
                return y 
            else :
                y = np.sin(omc*i)/(i*np.pi)
                return y 
        omc = (2*np.pi*fcd) *self.fs

        yn = np.zeros(round(self.N/Ndw))
        #audio Global 

        for n in range(round(self.N/Ndw)):
            for m in range(Ndw):
                yn[n] += lpffir(omc,m) * self.audio1[n*Ndw-m] 
        self.fs = self.fs / Ndw
        cutoff_frequency = 250
        sampling_period = 1/self.fs
        orde=2

        y = np.zeros(len(yn))  # Initialize the output signal
        omega_c = 2 * np.pi * cutoff_frequency
        omega_c_squared = omega_c*omega_c
        sampling_period_squared = sampling_period*sampling_period

        for n in range(2, len(yn)):
            y[n] = (((8/sampling_period_squared)-2*omega_c_squared) * y[n-1]
                    - ((4/sampling_period_squared) - (2 * np.sqrt(2) * omega_c / sampling_period) + omega_c_squared) * y[n-2]
                    + omega_c_squared * yn[n]
                    + 2 * omega_c_squared * yn[n-1]
                    + omega_c_squared * yn[n-2]) / ((4/sampling_period_squared) + (2 * np.sqrt(2) * omega_c / sampling_period) + omega_c_squared)

        self.filtered_lowpassMAV = y
        self.a1 = np.arange(len(self.filtered_lowpassMAV))
        self.widget.canvas.axes.clear()
        self.widget.canvas.axes.set_title('Signal')
        self.widget.canvas.axes.set_xlabel('time')
        self.widget.canvas.axes.set_ylabel('Amplitude')
        self.widget.canvas.axes.plot(self.a1/self.fs,self.filtered_lowpassMAV,label="Proces")
        self.widget.canvas.axes.legend()
        self.widget.canvas.axes.grid()
        self.widget.canvas.draw()


    def env (self):
        Ht = np.zeros(len(self.filtered_lowpassMAV))
        eh = np.zeros(len(self.filtered_lowpassMAV))
        for t in range(len(self.filtered_lowpassMAV)):
            for ta in range(len(self.filtered_lowpassMAV)):
                if (ta-t) == 0:
                    continue
                else:
                    Ht[t] += self.filtered_lowpassMAV[ta] /(ta-t)

        for x in range(len(self.filtered_lowpassMAV)):
            eh[x] = np.sqrt(np.square(self.filtered_lowpassMAV[x]) + np.square(Ht[x]))
        sum = 0
        window=50
        mAver = []
        k = int((window-1)/2)
        for i in np.arange(k, len(eh)-k):
            for ii in np.arange(i-k, i+k):
                sum = sum + eh[ii]
            mAver.append(sum / window)
            sum = 0
        zeros = [0]*k
        mAver = zeros + mAver + zeros
        self.maver = mAver
        self.a1 = np.arange(len(self.filtered_lowpassMAV))
        self.widget.canvas.axes.clear()
        self.widget.canvas.axes.set_title('Signal')
        self.widget.canvas.axes.set_xlabel('time')
        self.widget.canvas.axes.set_ylabel('Amplitude')
        self.widget.canvas.axes.plot(self.a1/2000,self.filtered_lowpassMAV,label="Proces")
        self.widget.canvas.axes.plot(self.a1/2000,self.maver,label="Envelope")
        self.widget.canvas.axes.legend()
        self.widget.canvas.axes.grid()
        self.widget.canvas.draw()
    def update (self,value):
        self.label_2.setText(str(value/100))
        th = np.zeros(len(self.filtered_lowpassMAV))
        for i in range(len(self.filtered_lowpassMAV)):
            th[i] = value/100
        self.widget.canvas.axes.clear()
        self.widget.canvas.axes.set_title('Signal')
        self.widget.canvas.axes.set_xlabel('time')
        self.widget.canvas.axes.set_ylabel('Amplitude')
        self.widget.canvas.axes.plot(self.a1/self.fs,self.filtered_lowpassMAV,label="Proces")
        self.widget.canvas.axes.plot(self.a1/self.fs,self.maver,label="Envelope")
        self.widget.canvas.axes.plot(self.a1/self.fs,th,label="Thresholding")
        self.widget.canvas.axes.legend()
        self.widget.canvas.axes.grid()
        self.widget.canvas.draw()   
        self.ind = (value/100)
    def thr(self):
        for t in range(len(self.maver)):
            if self.maver[t] < self.ind: 
                self.maver[t] = 0
            else :
                self.maver[t] =0.5
        self.widget.canvas.axes.clear()
        self.widget.canvas.axes.set_title('Signal')
        self.widget.canvas.axes.set_xlabel('time')
        self.widget.canvas.axes.set_ylabel('Amplitude')
        self.widget.canvas.axes.plot(self.a1/self.fs,self.filtered_lowpassMAV,label="Proces")
        self.widget.canvas.axes.plot(self.a1/self.fs,self.maver,label="Envelope")
        self.widget.canvas.axes.legend()
        self.widget.canvas.axes.grid()
        self.widget.canvas.draw()     
    def hsl (self):
        aw = []
        bl = []
        jml = 0 

        for n in range(len(self.maver)):
            if (self.maver[n]==0.5)and(self.maver[n-1]==0):
                aw.append(n)
            if (self.maver[n]==0) and (self.maver[n-1]==0.5):
                bl.append(n)
                jml+=1
        aw = np.array(aw)
        bl = np.array(bl)

        # length
        s1l = np.zeros(int(np.ceil(jml/2)))
        s2l = np.zeros(int(np.ceil(jml/2)))
        sysl = np.zeros(int(np.ceil(jml/2)))
        dysl = np.zeros(int(np.ceil(jml/2)))

        jh = 0
        for i in range(jml):
            if i % 2 == 0:
                s1l[jh] = np.abs(aw[i]-bl[i])
            else :
                s2l[jh] = np.abs(aw[i]-bl[i])
                jh+= 1

        jh = 0
        for i in range(jml):
            if i == 0 :
                continue
            if i % 2 == 1:
                sysl[jh] = aw[i] - bl[i-1]
            if i%2 == 0:
                dysl[jh] = aw[i] - bl[i-1]
                jh +=1
        dysl[(int(np.ceil(jml/2)))-1] = np.abs(len(self.maver) - bl[i])
        s1l = s1l/self.fs
        s2l = s2l/self.fs 
        dysl = dysl/self.fs
        sysl = sysl/self.fs 

        msys = np.mean(sysl)
        mdys = np.mean(dysl)
        ms1 = np.mean(s1l)
        jh = 0
        s1d = np.zeros(int(np.ceil(jml/2)))
        s2d = np.zeros(int(np.ceil(jml/2)))
        sysd = np.zeros(int(np.ceil(jml/2)))
        dysd = np.zeros(int(np.ceil(jml/2)))
        for i in range (jml):
            if i % 2 == 0:
                s1p = self.filtered_lowpassMAV[aw[i]:bl[i]]
                N = len(s1p)
                n = np.arange(0,N,1,dtype=int)
                k = np.arange(0,N,1,dtype=int) 
                yf = fft(s1p) 
                for n in range(N//2):
                    s1d[jh] += np.abs(yf[n])
            else :
                s2p = self.filtered_lowpassMAV[aw[i]:bl[i]]
                N = len(s2p)
                n = np.arange(0,N,1,dtype=int)
                k = np.arange(0,N,1,dtype=int) 
                yf = fft(s2p) 
                for n in range(N//2):
                    s2d[jh] += np.abs(yf[n])
                jh+= 1  
        jh = 0
        for i in range(jml):
            if i == 0 :
                continue
            if i % 2 == 1:
                sysp = self.filtered_lowpassMAV[bl[i-1]:aw[i]]
                N = len(sysp)
                n = np.arange(0,N,1,dtype=int)
                k = np.arange(0,N,1,dtype=int) 
                yf = fft(sysp) 
                for n in range(N//2):
                    sysd[jh] += np.abs(yf[n])          
            if i%2 == 0:
                dysp = self.filtered_lowpassMAV[bl[i-1]:aw[i]]
                N = len(dysp)
                n = np.arange(0,N,1,dtype=int)
                k = np.arange(0,N,1,dtype=int) 
                yf = fft(dysp) 
                for n in range(N//2):
                    dysd[jh] += np.abs(yf[n])
                jh +=1 
        dysp = self.filtered_lowpassMAV[bl[i]:len(self.maver)]
        N = len(dysp)
        n = np.arange(0,N,1,dtype=int)
        k = np.arange(0,N,1,dtype=int) 
        yf = fft(dysp) 
        for n in range(N//2):
            dysd[(int(np.ceil(jml/2)))-1] += np.abs(yf[n]) 
        mean_sysd = np.mean(sysd)
        mean_dysd = np.mean(dysd) 
        df_msys = pd.DataFrame({'Mean Sys': [msys]})
        df_mdys = pd.DataFrame({'Mean Dys': [mdys]})
        df_meansysd = pd.DataFrame({'F Mean Sys': [mean_sysd]})
        df_meandysd = pd.DataFrame({'F Mean Dys': [mean_dysd]})
        df_ms1 = pd.DataFrame({'Mean S1': [ms1]})
        df_result = pd.concat([ 
                            df_msys, 
                            df_mdys,  
                            df_meansysd,
                            df_meandysd,
                            ], axis=1)
        df1=pd.DataFrame(df_result)
        data=pd.read_excel('9.xlsx')
        Y = data['0/1']
        X = data.drop(['0/1'], axis = 1)
        x_training, x_testing, y_training, y_testing = train_test_split(X, Y, test_size = 0.2, random_state = 0)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_training,y_training)
        prediction = knn.predict(x_testing)
        prediction = knn.predict(df1)
        if prediction[0] == 1 :
            self.lineEdit_3.setText("Normal")
        else : 
            self.lineEdit_3.setText("Abnormal")


app = QApplication([])
window = widgetss()
window.show()
app.exec_()