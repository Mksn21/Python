from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
import pandas as pd
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import numpy as np
import pandas as pd 
import pyhrv
import matplotlib as mpl
from scipy.fft import fft, fftfreq

def dirac(x):
    if (x ==0):
        dirac_delta = 1
    else:
        dirac_delta = 0 
    return dirac_delta

class widgetss (QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("a.ui",self)
        self.pushButton.clicked.connect(self.insp)
        self.pushButton_2.clicked.connect(self.Dwt)
        self.pushButton_3.clicked.connect(self.Hrv)
        self.pushButton_4.clicked.connect(self.Ftr)
        self.pushButton_5.clicked.connect(self.RRT)

    def insp(self):
        try :
            self.widget.canvas.axes.clear()
            df = QFileDialog.getOpenFileName(self,'Open File',"",'All Files (*);;Txt Files(*.txt)')
            df = pd.read_csv(df[0],sep='\s+',header= None)
            df=pd.DataFrame(df)
            self.x = np.arange(len(df))/125
            self.y = np.array(df[df.columns[0]])/1e8
            self.widget.canvas.axes.plot(self.x[0:2000],self.y[0:2000],color="b")
            self.widget.canvas.axes.set_title('Raw ECG Data')
            self.widget.canvas.axes.set_xlabel('Time (s)')
            self.widget.canvas.axes.set_ylabel('Amplitude (mV)')
            self.widget.canvas.axes.grid()
            self.widget.canvas.draw()
            self.fs=int(round(1/(self.x[1]-self.x[0])))
            self.lineEdit.setText(str(self.fs))
            self.jumlahdata = int(np.size(self.x))
            self.lineEdit_2.setText(str(self.jumlahdata))
        except : 
            pass
    def Dwt (self):
        self.widget_3.canvas.axes.clear()
        self.widget_2.canvas.axes.clear()
        self.widget_4.canvas.axes.clear()
        self.widget_5.canvas.axes.clear()
        self.widget_11.canvas.axes.clear()
        self.widget_6.canvas.axes.clear()
        self.widget_7.canvas.axes.clear()
        self.widget_8.canvas.axes.clear()
        self.widget_9.canvas.axes.clear()
        self.widget_10.canvas.axes.clear()
        self.h = []
        self.g = []
        self.n_list = []

        for n in range(-2,2):
            self.n_list.append(n)
            temp_h = 1/8 * (dirac(n-1) + 3*dirac(n) + 3*dirac(n+1) + dirac(n+2))
            self.h.append(temp_h)
            temp_g = -2 * (dirac(n)-dirac(n+1))
            self.g.append(temp_g)
        self.Hw = np.zeros(20000)
        self.Gw = np.zeros(20000)
        self.i_list = []

        for i in range(0,self.fs+1):
            self.i_list.append(i)
            reG = 0
            imG = 0 
            reH = 0
            imH = 0 
            for k in range(-2,2):
                reG += self.g[k+2] * np.cos(k*2*np.pi*i/self.fs)
                imG -= self.g[k+2] * np.sin(k*2*np.pi*i/self.fs) 
                reH += self.h[k+2] * np.cos(k*2*np.pi*i/self.fs)
                imH -= self.h[k+2] * np.sin(k*2*np.pi*i/self.fs)
            self.Hw[i] = np.sqrt((reH**2) + (imH**2))
            self.Gw[i] = np.sqrt((reG**2) + (imG**2))

        self.i_list = self.i_list[0:round(self.fs/2)]

        self.widget_2.canvas.axes.bar(self.n_list,self.h,0.1)
        self.widget_2.canvas.axes.set_title('Koefisien Filter h(n)')
        self.widget_2.canvas.axes.set_xlabel('n')
        self.widget_2.canvas.axes.set_ylabel('h(n)')
        self.widget_2.canvas.axes.grid()
        self.widget_2.canvas.draw()

        self.widget_3.canvas.axes.bar(self.n_list,self.g,0.1)
        self.widget_3.canvas.axes.set_title('Koefisien Filter g(n)')
        self.widget_3.canvas.axes.set_xlabel('n')
        self.widget_3.canvas.axes.set_ylabel('g(n)')
        self.widget_3.canvas.axes.grid()
        self.widget_3.canvas.draw()

        self.widget_4.canvas.axes.plot(self.i_list,self.Hw[0:len(self.i_list)])
        self.widget_4.canvas.axes.set_title('Frekuensi Respone H(f)')
        self.widget_4.canvas.axes.set_xlabel('f')
        self.widget_4.canvas.axes.set_ylabel('H(f)')
        self.widget_4.canvas.axes.grid()
        self.widget_4.canvas.draw()

        self.widget_5.canvas.axes.plot(self.i_list,self.Gw[0:len(self.i_list)])
        self.widget_5.canvas.axes.set_title('Frekuensi Respone G(f)')
        self.widget_5.canvas.axes.set_xlabel('f')
        self.widget_5.canvas.axes.set_ylabel('G(f)')
        self.widget_5.canvas.axes.grid()
        self.widget_5.canvas.draw()

        self.Q = np.zeros((9,round(self.fs/2)+1))
        self.i_list = []
        for i in range(0,round(self.fs/2)+1):
            self.i_list.append(i)
            self.Q[1][i] = self.Gw[i]
            self.Q[2][i] = self.Gw[2*i]*self.Hw[i]
            self.Q[3][i] = self.Gw[4*i]*self.Hw[2*i]*self.Hw[i]
            self.Q[4][i] = self.Gw[8*i]*self.Hw[4*i]*self.Hw[2*i]*self.Hw[i]
            self.Q[5][i] = self.Gw[16*i]*self.Hw[8*i]*self.Hw[4*i]*self.Hw[2*i]*self.Hw[i]
            self.Q[6][i] = self.Gw[32*i]*self.Hw[16*i]*self.Hw[8*i]*self.Hw[4*i]*self.Hw[2*i]*self.Hw[i]
            self.Q[7][i] = self.Gw[64*i]*self.Hw[32*i]*self.Hw[16*i]*self.Hw[8*i]*self.Hw[4*i]*self.Hw[2*i]*self.Hw[i]
            self.Q[8][i] = self.Gw[128*i]*self.Hw[64*i]*self.Hw[32*i]*self.Hw[16*i]*self.Hw[8*i]*self.Hw[4*i]*self.Hw[2*i]*self.Hw[i]
        for i in range(1,9):
            line_label = "Q{}".format(i)
            self.widget_11.canvas.axes.plot(self.i_list,self.Q[i],label = line_label)
        self.widget_11.canvas.axes.legend()
        self.widget_11.canvas.axes.set_title('Frekuensi Respone')
        self.widget_11.canvas.axes.set_xlabel('f')
        self.widget_11.canvas.axes.set_ylabel('Q(f)') 
        self.widget_11.canvas.draw() 

        self.qj = np.zeros((6,100000))
        k_list = []
        j = 1

        a = -(round(2**j) + round(2**(j-1))-2)
        print("a = ",a)

        b = -(1-round(2**(j-1))) + 1
        print("b =",b)

        for k in range(a,b):
            k_list.append(k)
            self.qj[1,k+abs(a)] = -2*(dirac(k)-dirac(k+1))

        self.widget_6.canvas.axes.bar(k_list,self.qj[1][0:len(k_list)])
        self.widget_6.canvas.axes.set_title('Koefisien Filter q1(n)')
        self.widget_6.canvas.axes.set_xlabel('n')
        self.widget_6.canvas.axes.set_ylabel('q1(n)')
        self.widget_6.canvas.draw() 

        k_list = []
        j = 2

        a = -(round(2**j) + round(2**(j-1))-2)
        print("a = ",a)

        b = -(1-round(2**(j-1))) + 1
        print("b =",b)

        for k in range(a,b):
            k_list.append(k)
            self.qj[2,k+abs(a)] = -1/4*(dirac(k-1)+3*dirac(k)+2*dirac(k+1)-2*dirac(k+2)
            -3*dirac(k+3)-dirac(k+4))

        self.widget_10.canvas.axes.bar(k_list,self.qj[2][0:len(k_list)])
        self.widget_10.canvas.axes.set_title('Koefisien Filter q2(n)')
        self.widget_10.canvas.axes.set_xlabel('n')
        self.widget_10.canvas.axes.set_ylabel('q2(n)')
        self.widget_10.canvas.draw() 

        k_list = []
        j = 3

        a = -(round(2**j) + round(2**(j-1))-2)
        print("a = ",a)

        b = -(1-round(2**(j-1))) + 1
        print("b =",b)

        for k in range(a,b):
            k_list.append(k)
            self.qj[3,k+abs(a)] = -1/32*(dirac(k-3)+3*dirac(k-2)+6*dirac(k-1)+10*dirac(k)
            +11*dirac(k+1)+9*dirac(k+2)+4*dirac(k+3)-4*dirac(k+4)-9*dirac(k+5)
            -11*dirac(k+6)-10*dirac(k+7)-6*dirac(k+8)-3*dirac(k+9)-dirac(k+10))

        self.widget_9.canvas.axes.bar(k_list,self.qj[3][0:len(k_list)])
        self.widget_9.canvas.axes.set_title('Koefisien Filter q3(n)')
        self.widget_9.canvas.axes.set_xlabel('n')
        self.widget_9.canvas.axes.set_ylabel('q3(n)')
        self.widget_9.canvas.draw() 

        k_list = []
        j = 4

        a = -(round(2**j) + round(2**(j-1))-2)
        print("a = ",a)

        b = -(1-round(2**(j-1))) + 1
        print("b =",b)

        for k in range(a,b):
            k_list.append(k)
            self.qj[4,k+abs(a)] = -1/256*(dirac(k-7)+3*dirac(k-6)+6*dirac(k-5)+10*dirac(k-4)+15*dirac(k-3)
            +21*dirac(k-2)+28*dirac(k-1)+36*dirac(k)+41*dirac(k+1)+43*dirac(k+2)
            +42*dirac(k+3)+38*dirac(k+4)+31*dirac(k+5)+21*dirac(k+6)+8*dirac(k+7)
            -8*dirac(k+8)-21*dirac(k+9)-31*dirac(k+10)-38*dirac(k+11)-42*dirac(k+12)
            -43*dirac(k+13)-41*dirac(k+14)-36*dirac(k+15)-28*dirac(k+16)-21*dirac(k+17)
            -15*dirac(k+18)-10*dirac(k+19)-6*dirac(k+20)-3*dirac(k+21) -dirac(k+22))


        self.widget_8.canvas.axes.bar(k_list,self.qj[4][0:len(k_list)])
        self.widget_8.canvas.axes.set_title('Koefisien Filter q4(n)')
        self.widget_8.canvas.axes.set_xlabel('n')
        self.widget_8.canvas.axes.set_ylabel('q4(n)')
        self.widget_8.canvas.draw() 

        k_list = []
        j = 5

        a = -(round(2**j) + round(2**(j-1))-2)
        print("a = ",a)

        b = -(1-round(2**(j-1))) + 1
        print("b =",b)

        for k in range(a,b):
            k_list.append(k)
            self.qj[5,k+abs(a)] = -1/(512)*(dirac(k-15)+3*dirac(k-14)+6*dirac(k-13)+10*dirac(k-12)+15*dirac(k-11)+21*dirac(k-10)
            +28*dirac(k-9)+36*dirac(k-8)+45*dirac(k-7)+55*dirac(k-6)+66*dirac(k-5)+78*dirac(k-4)
            +91*dirac(k-3)+105*dirac(k-2)+120*dirac(k-1)+136*dirac(k)+149*dirac(k+1)+159*dirac(k+2)
            +166*dirac(k+3)+170*dirac(k+4)+171*dirac(k+5)+169*dirac(k+6)+164*dirac(k+7)+156*dirac(k+8)
            +145*dirac(k+9)+131*dirac(k+10)+114*dirac(k+11)+94*dirac(k+12)+71*dirac(k+13)+45*dirac(k+14)
            +16*dirac(k+15)-16*dirac(k+16)-45*dirac(k+17)-71*dirac(k+18)-94*dirac(k+19)-114*dirac(k+20)
            -131*dirac(k+21)-145*dirac(k+22)-156*dirac(k+23)-164*dirac(k+24)-169*dirac(k+25)
            -171*dirac(k+26)-170*dirac(k+27)-166*dirac(k+28)-159*dirac(k+29)-149*dirac(k+30)
            -136*dirac(k+31)-120*dirac(k+32)-105*dirac(k+33)-91*dirac(k+34)-78*dirac(k+35)
            -66*dirac(k+36)-55*dirac(k+37)-45*dirac(k+38)-36*dirac(k+39)-28*dirac(k+40)
            -21*dirac(k+41)-15*dirac(k+42)-10*dirac(k+43)-6*dirac(k+44)-3*dirac(k+45)
            -dirac(k+46))

        self.widget_7.canvas.axes.bar(k_list,self.qj[5][0:len(k_list)])
        self.widget_7.canvas.axes.set_title('Koefisien Filter q5(n)')
        self.widget_7.canvas.axes.set_xlabel('n')
        self.widget_7.canvas.axes.set_ylabel('q5(n)')
        self.widget_7.canvas.draw() 

        self.w2fm = np.zeros((5,self.jumlahdata))
        self.s2fm = np.zeros((5,self.jumlahdata))

        for n in range(self.jumlahdata):
            for j in range(1,6):
                for k in range(-2,2):
                    try:
                        self.w2fm[j-1,n] += self.g[k+2] * self.y[round(n-np.power(2,j-1)*k)]
                        self.s2fm[j-1,n] += self.h[k+2] * self.y[round(n-np.power(2,j-1)*k)]
                    except:
                        self.w2fm[j-1,n] += 0 
                        self.s2fm[j-1,n] += 0
        n = np.arange(2000)
        self.widget_12.canvas.axes.plot(n/self.fs,self.w2fm[0,n])
        self.widget_12.canvas.axes.set_title('w2fm1')
        self.widget_12.canvas.axes.set_xlabel('Time (s)')
        self.widget_12.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_12.canvas.axes.grid()
        self.widget_12.canvas.draw()

        self.widget_13.canvas.axes.plot(n/self.fs,self.s2fm[0,n])
        self.widget_13.canvas.axes.set_title('s2fm1')
        self.widget_13.canvas.axes.set_xlabel('Time (s)')
        self.widget_13.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_13.canvas.axes.grid()
        self.widget_13.canvas.draw()

        self.widget_14.canvas.axes.plot(n/self.fs,self.w2fm[1,n])
        self.widget_14.canvas.axes.set_title('w2fm2')
        self.widget_14.canvas.axes.set_xlabel('Time (s)')
        self.widget_14.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_14.canvas.axes.grid()
        self.widget_14.canvas.draw()

        self.widget_15.canvas.axes.plot(n/self.fs,self.s2fm[1,n])
        self.widget_15.canvas.axes.set_title('s2fm2')
        self.widget_15.canvas.axes.set_xlabel('Time (s)')
        self.widget_15.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_15.canvas.axes.grid()
        self.widget_15.canvas.draw()

        self.widget_16.canvas.axes.plot(n/self.fs,self.w2fm[2,n])
        self.widget_16.canvas.axes.set_title('w2fm3')
        self.widget_16.canvas.axes.set_xlabel('Time (s)')
        self.widget_16.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_16.canvas.axes.grid()
        self.widget_16.canvas.draw()

        self.widget_17.canvas.axes.plot(n/self.fs,self.s2fm[2,n])
        self.widget_17.canvas.axes.set_title('s2fm3')
        self.widget_17.canvas.axes.set_xlabel('Time (s)')
        self.widget_17.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_17.canvas.axes.grid()
        self.widget_17.canvas.draw()

        self.widget_18.canvas.axes.plot(n/self.fs,self.w2fm[3,n])
        self.widget_18.canvas.axes.set_title('w2fm4')
        self.widget_18.canvas.axes.set_xlabel('Time (s)')
        self.widget_18.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_18.canvas.axes.grid()
        self.widget_18.canvas.draw()

        self.widget_19.canvas.axes.plot(n/self.fs,self.s2fm[3,n])
        self.widget_19.canvas.axes.set_title('s2fm4')
        self.widget_19.canvas.axes.set_xlabel('Time (s)')
        self.widget_19.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_19.canvas.axes.grid()
        self.widget_19.canvas.draw()

        self.widget_20.canvas.axes.plot(n/self.fs,self.w2fm[4,n])
        self.widget_20.canvas.axes.set_title('w2fm5')
        self.widget_20.canvas.axes.set_xlabel('Time (s)')
        self.widget_20.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_20.canvas.axes.grid()
        self.widget_20.canvas.draw()

        self.widget_21.canvas.axes.plot(n/self.fs,self.s2fm[4,n])
        self.widget_21.canvas.axes.set_title('s2fm5')
        self.widget_21.canvas.axes.set_xlabel('Time (s)')
        self.widget_21.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_21.canvas.axes.grid()
        self.widget_21.canvas.draw()

        self.T1 = round(2**(1-1)) - 1
        self.T2 = round(2**(2-1)) - 1
        self.T3 = round(2**(3-1)) - 1
        self.T4 = round(2**(4-1)) - 1
        self.T5 = round(2**(5-1)) - 1

        self.w2fb = np.zeros((6,self.jumlahdata+self.T5))
        for n in range(self.jumlahdata):
            for j in range(1,6):
                self.w2fb[1][n+self.T1]=0
                self.w2fb[2][n+self.T2]=0
                self.w2fb[3][n+self.T3]=0
                self.w2fb[4][n+self.T4]=0
                self.w2fb[5][n+self.T5]=0

                a = -(round(2**j)+round(2**(j-1))-2)
                b = -(1-round(2**(j-1)))
                for k in range(a,b+1):
                        self.w2fb[1][n+self.T1] += self.qj[1,(k+abs(a))]*self.y[n-(k+abs(a))]
                        self.w2fb[2][n+self.T2] += self.qj[2,(k+abs(a))]*self.y[n-(k+abs(a))] 
                        self.w2fb[3][n+self.T3] += self.qj[3,(k+abs(a))]*self.y[n-(k+abs(a))]
                        self.w2fb[4][n+self.T4] += self.qj[4,(k+abs(a))]*self.y[n-(k+abs(a))] 
                        self.w2fb[5][n+self.T5] += self.qj[5,(k+abs(a))]*self.y[n-(k+abs(a))] 

        n = np.arange(1000)
        self.widget_22.canvas.axes.plot(n/self.fs,self.w2fb[1][0:len(n)])
        self.widget_22.canvas.axes.set_title('w2fb1')
        self.widget_22.canvas.axes.set_xlabel('Time (s)')
        self.widget_22.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_22.canvas.axes.grid()
        self.widget_22.canvas.draw()

        self.widget_24.canvas.axes.plot(n/self.fs,self.w2fb[2][0:len(n)])
        self.widget_24.canvas.axes.set_title('w2fb2')
        self.widget_24.canvas.axes.set_xlabel('Time (s)')
        self.widget_24.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_24.canvas.axes.grid()
        self.widget_24.canvas.draw()

        self.widget_26.canvas.axes.plot(n/self.fs,self.w2fb[3][0:len(n)])
        self.widget_26.canvas.axes.set_title('w2fb3')
        self.widget_26.canvas.axes.set_xlabel('Time (s)')
        self.widget_26.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_26.canvas.axes.grid()
        self.widget_26.canvas.draw()

        self.widget_37.canvas.axes.plot(n/self.fs,self.w2fb[5][0:len(n)])
        self.widget_37.canvas.axes.set_title('w2fb5')
        self.widget_37.canvas.axes.set_xlabel('Time (s)')
        self.widget_37.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_37.canvas.axes.grid()
        self.widget_37.canvas.draw()

        self.widget_35.canvas.axes.plot(n/self.fs,self.w2fb[4][0:len(n)])
        self.widget_35.canvas.axes.set_title('w2fb4')
        self.widget_35.canvas.axes.set_xlabel('Time (s)')
        self.widget_35.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_35.canvas.axes.grid()
        self.widget_35.canvas.draw()


    def Hrv (self):
        self.gradien3 = np.zeros(self.jumlahdata)
        delay3 = self.T3

        for n in range(delay3,self.jumlahdata):
            self.gradien3[n] = self.w2fb[3][n-delay3] - self.w2fb[3][n+delay3]
        n = np.arange(self.jumlahdata)
        self.widget_23.canvas.axes.plot(n[0:1000]/self.fs,self.gradien3[0:1000])
        self.widget_23.canvas.axes.set_title('Gradien Level 3')
        self.widget_23.canvas.axes.set_xlabel('Time (s)')
        self.widget_23.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_23.canvas.axes.grid()
        self.widget_23.canvas.draw()

        self.hasilQRS = np.zeros(self.jumlahdata)

        for n in range(self.jumlahdata):
            if self.gradien3[n] > 1.5:
                self.hasilQRS[n-(self.T4+1)] = 5
            else :
                self.hasilQRS[n-(self.T4+1)] = 0 

        self.widget_25.canvas.axes.plot(self.x[0:2000],self.y[0:2000],color="b")
        self.widget_25.canvas.axes.plot(self.x[0:2000],self.hasilQRS[0:2000],color="r")
        self.widget_25.canvas.axes.set_title('Gradien Level 3')
        self.widget_25.canvas.axes.set_xlabel('Time (s)')
        self.widget_25.canvas.axes.set_ylabel('Amplitude (mV)')
        self.widget_25.canvas.axes.grid()
        self.widget_25.canvas.draw()

        self.ptp = 0 
        self.waktu = np.zeros(np.size(self.hasilQRS))
        self.selisih = np.zeros(np.size(self.hasilQRS))
        for n in range(np.size(self.hasilQRS)-1):
            if self.hasilQRS[n] <self.hasilQRS[n+1]:
                self.waktu[self.ptp] = n/self.fs 
                self.selisih[self.ptp] = self.waktu[self.ptp] -self.waktu[self.ptp-1]
                self.ptp +=1
        self.ptp = self.ptp-1

        j = 0 
        self.peak = np.zeros(np.size(self.hasilQRS))
        for n in range(np.size(self.hasilQRS)):
            if (self.hasilQRS[n]==5) and (self.hasilQRS[n-1]==0):
                self.peak[j] = n
                j+=1
        temp = 0
        self.interval = np.zeros(np.size(self.hasilQRS))
        self.BPM = np.zeros(np.size(self.hasilQRS))
        for n in range(self.ptp):
            self.interval[n] = (self.peak[n]-self.peak[n-1])*(1/self.fs)
            self.BPM[n] = 60/self.interval[n]
            temp = temp+ self.BPM[n]
            self.rata = temp/(n-1)
        self.rata

        self.bpm_rr = np.zeros(self.ptp)
        self.rr_t = np.zeros(self.ptp)
        for n in range(self.ptp):
            self.bpm_rr[n] = 60/self.selisih[n]
            self.rr_t[n] = self.selisih[n]
            if self.bpm_rr[n] > 100:
                self.bpm_rr[n] = self.rata

        self.mean_rr = np.sum(self.rr_t)/len(self.rr_t)
        self.fs_hrv = 1/self.mean_rr

        n = np.arange(self.ptp)

        self.widget_27.canvas.axes.plot(n,self.bpm_rr,color="red")
        self.widget_27.canvas.axes.set_title('Tachogram')
        self.widget_27.canvas.axes.set_xlabel('Sequence (n)')
        self.widget_27.canvas.axes.set_ylabel('bpm')
        self.widget_27.canvas.axes.grid()
        self.widget_27.canvas.draw()

    def Ftr (self):

        
        self.widget_30.canvas.axes.hist(self.bpm_rr,bins=self.ptp,color="red")
        self.widget_30.canvas.axes.set_title('Histogram')
        self.widget_30.canvas.axes.set_xlabel('bpm')
        self.widget_30.canvas.axes.set_ylabel('n')
        self.widget_30.canvas.axes.set_ylim(0,10)
        self.widget_30.canvas.axes.grid()
        self.widget_30.canvas.draw()

        RR_SDNN = 0

        for n in range(self.ptp):
            RR_SDNN += (((self.rr_t[n])-(self.mean_rr))**2)

        SDNN = np.sqrt(RR_SDNN/(self.ptp-1))
        self.lineEdit_3.setText(str(SDNN))

        RR_RMSSD = 0
        for n in range(self.ptp):
            RR_RMSSD += ((self.selisih[n+1]-self.selisih[n])**2)

        RMSSD = np.sqrt(RR_RMSSD/(self.ptp-1))
        self.lineEdit_4.setText(str(RMSSD))

        NN50 = 0 

        for n in range(self.ptp):
            if (abs(self.selisih[n+1]-self.selisih[n]) > 0.05):
                NN50 += 1
        pNN50 = (NN50/(self.ptp-1))*100
        self.lineEdit_5.setText(str(pNN50))

        dif = 0

        for n in range(self.ptp):
            dif += abs(self.selisih[n]-self.selisih[n+1])
        RRdif = dif/(self.ptp-1)
        RR_SDSD = 0 

        for n in range(self.ptp):
            RR_SDSD += (((abs(self.selisih[n]-self.selisih[n+1]))-RRdif)**2)

        SDSD = np.sqrt(RR_SDSD/(self.ptp-2))
        self.lineEdit_6.setText(str(SDSD))

        M=20
        w=np.zeros(M)
        for n in range(M-1):
            w[n]=0.54-0.46*np.cos((2*n*np.pi)/M)
        bpm_rrn = self.bpm_rr - (np.sum(self.bpm_rr)/len(self.bpm_rr))
        T_bpm = 1/self.fs_hrv
        N = self.ptp 
        array_bpm = np.zeros(self.ptp)
        self.array_bpmj = np.zeros(self.ptp)
        sinyal_bpm = np.zeros(self.ptp)
        nn = 0
        while True :
            window_ptp = np.zeros(self.ptp)
            jj = 0 
            for i in range(round((M/2)*nn),round(((M/2)*nn))+M):
                if i >= self.ptp:
                    continue
                else:
                    window_ptp[i] = w[jj]
                    jj += 1
            
            for i in range(self.ptp):
                sinyal_bpm[i] = bpm_rrn[i] * window_ptp[i]

            array_bpm1 = fft(sinyal_bpm)
            array_bpm2 = np.abs(array_bpm1)
            for i in range(self.ptp):
                array_bpm[i] += array_bpm2[i] 
                self.array_bpmj[i] += array_bpm1[i]
            
            nn += 1

            if round((M/2)*nn)> self.ptp :
                break
        
        k = np.arange(0,self.ptp//2,1,dtype=int) 
        array_bpm = array_bpm/nn

        self.VLH_PSD = np.zeros(self.ptp//2)
        self.LF_PSD = np.zeros(self.ptp//2)
        self.HF_PSD = np.zeros(self.ptp//2)
        self.TLP_PSD = np.zeros(self.ptp//2)

        for i in range(self.ptp//2):
            if i*self.fs_hrv/self.ptp > 0.003 and i*self.fs_hrv/self.ptp < 0.04:
                self.VLH_PSD[i] = array_bpm[i]
            elif i*self.fs_hrv/self.ptp >= 0.04 and i*self.fs_hrv/self.ptp < 0.15:
                self.LF_PSD[i] = array_bpm[i]
            elif i*self.fs_hrv/self.ptp > 0.15 and i*self.fs_hrv/self.ptp < 0.4:
                self.HF_PSD[i] = array_bpm[i]

        for i in range(self.ptp//2):
            if i*self.fs_hrv/self.ptp < 0.4:
                self.TLP_PSD[i] = array_bpm[i]
        
        self.widget_31.canvas.axes.fill_between(k*self.fs_hrv/self.ptp,array_bpm[0:self.ptp//2],where=k*self.fs_hrv/self.ptp < 0.4,alpha=0.8,linewidth = 5)
        self.widget_31.canvas.axes.annotate("VLH",(0.015,20),size=12 ,bbox = dict(boxstyle="round",color="green",pad = 0.8),color="white",weight="bold")
        self.widget_31.canvas.axes.annotate("LF",(0.09,20),size=12 ,bbox = dict(boxstyle="round",color="green",pad = 0.8),color="white",weight="bold")
        self.widget_31.canvas.axes.annotate("HF",(0.27,20),size=12 ,bbox = dict(boxstyle="round",color="green",pad = 0.8),color="white",weight="bold")
        self.widget_31.canvas.axes.axvline(x=0.04,color="#D3D3D3", linestyle="--")
        self.widget_31.canvas.axes.axvline(x=0.15,color="#D3D3D3", linestyle="--")
        self.widget_31.canvas.axes.axvline(x=0.4,color="#D3D3D3", linestyle="--")
        self.widget_31.canvas.axes.set_title("Welch Method")
        self.widget_31.canvas.axes.set_xlabel("Freq (Hz)")
        self.widget_31.canvas.axes.set_ylabel("PSD")
        self.widget_31.canvas.draw()     

        TLP = 0
        VLH_P = 0 
        LF_P = 0 
        HF_P = 0  

        for i in range(self.ptp//2):
            TLP += self.TLP_PSD[i]
            VLH_P += self.VLH_PSD[i]
            LF_P += self.LF_PSD[i]
            HF_P += self.HF_PSD[i]
            

        LF = (LF_P) /(TLP-VLH_P)
        HF = (HF_P) / (TLP-VLH_P)
        Ratio_LF_HF = LF/HF

        self.lineEdit_7.setText(str(LF))
        self.lineEdit_8.setText(str(HF))
        self.lineEdit_9.setText(str(Ratio_LF_HF))

        self.widget_32.canvas.axes.axhline(y=33.3, color = 'r', linestyle = '-')
        self.widget_32.canvas.axes.scatter(LF*100,HF*100)
        self.widget_32.canvas.axes.axhline(y=66.6, color = 'r', linestyle = '-')
        self.widget_32.canvas.axes.axvline(x=33.3, color = 'r', linestyle = '-')
        self.widget_32.canvas.axes.axvline(x=66.6, color = 'r', linestyle = '-')
        self.widget_32.canvas.axes.set_title("Autonomic Balance Diagram")
        self.widget_32.canvas.axes.set_xlabel("Sympathetic NS - LF")
        self.widget_32.canvas.axes.set_ylabel("Parasympathetic NS - LF")
        self.widget_32.canvas.axes.set_ylim(0,100)
        self.widget_32.canvas.axes.set_xlim(0,100)
        self.widget_32.canvas.draw()

        SD1_B = np.sqrt((np.square(SDSD*1000))/2)
        SD2_B =  np.sqrt((2*((SDNN*1000)**2))-(((SDSD*1000)**2)/2))
        SD_Ratio_B = SD2_B / SD1_B

        self.lineEdit_10.setText(str(SD1_B))
        self.lineEdit_11.setText(str(SD2_B))
        self.lineEdit_12.setText(str(SD_Ratio_B))
        RR_stl = np.zeros(self.ptp)
        for i in range(1,self.ptp):
            RR_stl[i-1] = self.rr_t[i]

        
        self.widget_33.canvas.axes.plot(self.rr_t*1000,RR_stl*1000, 'r%s' % 'o', markersize=2, alpha=0.5, zorder=3)
        self.widget_33.canvas.axes.set_title('$Poincar\acute{e}$')
        self.widget_33.canvas.axes.set_ylabel('$NNI_{i+1}$ [ms]')
        self.widget_33.canvas.axes.grid()
        ellipse_ = mpl.patches.Ellipse((self.mean_rr*1000, self.mean_rr*1000), SD1_B * 2, SD2_B * 2, angle=-45, fc='k', zorder=1)
        self.widget_33.canvas.axes.add_artist(ellipse_)
        ellipse_ = mpl.patches.Ellipse((self.mean_rr*1000, self.mean_rr*1000), SD1_B * 2 - 1, SD2_B * 2 - 1, angle=-45, fc='lightyellow', zorder=1)
        self.widget_33.canvas.axes.add_artist(ellipse_)
        arrow_head_size = 3
        na = 4
        a2 = self.widget_33.canvas.axes.arrow(
				self.mean_rr*1000, self.mean_rr*1000, (SD2_B - na) * np.cos(np.deg2rad(45)), (SD2_B - na) * np.sin(np.deg2rad(45)),
				head_width=arrow_head_size, head_length=arrow_head_size, fc='b', ec='b', zorder=4, linewidth=1.5)
        a1 = self.widget_33.canvas.axes.arrow(
				self.mean_rr*1000, self.mean_rr*1000, (-SD1_B + na) * np.cos(np.deg2rad(45)), (SD1_B - na) * np.sin(np.deg2rad(45)),
				head_width=arrow_head_size, head_length=arrow_head_size, fc='g', ec='g', zorder=4, linewidth=1.5)
        a3 = mpl.patches.Patch(facecolor='white', alpha=0.0)
        a4 = mpl.patches.Patch(facecolor='white', alpha=0.0)
        self.widget_33.canvas.axes.add_line(mpl.lines.Line2D(
				(min(self.rr_t*1000), max(self.rr_t*1000)),
				(min(self.rr_t*1000), max(self.rr_t*1000)),
				c='b', ls=':', alpha=0.6))
        self.widget_33.canvas.axes.add_line(mpl.lines.Line2D(
				(self.mean_rr*1000 - SD1_B * np.cos(np.deg2rad(45)) * na, self.mean_rr*1000 + SD1_B * np.cos(np.deg2rad(45)) * na),
				(self.mean_rr*1000 + SD1_B * np.sin(np.deg2rad(45)) * na, self.mean_rr*1000 - SD1_B * np.sin(np.deg2rad(45)) * na),
				c='g', ls=':', alpha=0.6))
        self.widget_33.canvas.axes.legend(
					[a1, a2, a3, a4],
					['SD1: %.3f$s$' % SD1_B, 'SD2: %.3f$s$' % SD2_B, 'SD1/SD2: %.3f' % (SD1_B/SD2_B)],
					framealpha=1)
        self.widget_33.canvas.draw()  
    def RRT (self):
        MPFx = 0
        MPFy = 0

        for i in range(self.ptp//2):
            MPFx += (i*self.fs_hrv/self.ptp) * self.HF_PSD[i]
            MPFy += self.HF_PSD[i]

        MPF = MPFx / MPFy 

        RRT = MPF * 60
        self.lineEdit_13.setText(str(MPF))
        self.lineEdit_14.setText(str(RRT))

        import scipy.fft
        HF_plt = np.zeros(self.ptp)

        for i in range(self.ptp):
            if i*self.fs_hrv/self.ptp > 0.15 and i*self.fs_hrv/self.ptp < 0.4:
                HF_plt[i] = self.array_bpmj[i]

        respi_signal = scipy.fft.ifft(HF_plt)
        respi_signal = np.abs(respi_signal)
        nnn = np.arange(len(respi_signal))

        self.widget_34.canvas.axes.plot(nnn/self.fs_hrv,respi_signal,color="red")
        self.widget_34.canvas.axes.set_title("Respiratory Signals")
        self.widget_34.canvas.axes.set_xlabel("Time (s)")
        self.widget_34.canvas.axes.set_ylabel("Amplitude")
        self.widget_34.canvas.draw()












        











        










app = QApplication([])
window = widgetss()
window.show()
app.exec_()