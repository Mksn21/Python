import sys
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from scipy import signal


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # -- INTERFACING -- #
        self.setGeometry(40, 80, 1850, 880)
        self.setWindowTitle("Signal Processing - DFT, STFT, CWT (ver. 01)")

        self.menu_bar = self.menuBar()
        self.menu_file = self.menu_bar.addMenu("File")
        self.menu_file_new = QtWidgets.QAction("New", self)
        self.menu_file.addAction(self.menu_file_new)
        self.menu_file_new.setEnabled(False)
        self.menu_file_open = QtWidgets.QAction("Open", self)
        self.menu_file.addAction(self.menu_file_open)
        self.menu_file.addSeparator()
        self.menu_file_exit = QtWidgets.QAction("Exit", self)
        self.menu_file.addAction(self.menu_file_exit)
        self.menu_file_exit.setEnabled(False)
        self.menu_setting = self.menu_bar.addMenu("Settings")
        self.menu_setting_signalProcessing = self.menu_setting.addMenu("Signal Processing")
        self.menu_setting_DFT = QtWidgets.QAction("DFT", self)
        self.menu_setting_signalProcessing.addAction(self.menu_setting_DFT)
        self.menu_setting_DFT.setEnabled(False)
        self.menu_setting_STFT = QtWidgets.QAction("STFT", self)
        self.menu_setting_signalProcessing.addAction(self.menu_setting_STFT)
        self.menu_setting_STFT.setEnabled(False)
        self.menu_setting_CWT = QtWidgets.QAction("CWT", self)
        self.menu_setting_signalProcessing.addAction(self.menu_setting_CWT)
        self.menu_setting_CWT.setEnabled(False)
        self.menu_help = self.menu_bar.addMenu("Help")
        self.menu_help.setEnabled(False)

        self.gb_1 = QtWidgets.QGroupBox("Input Signal", self)
        self.gb_1.setGeometry(20, 50, 1200, 300)
        self.cv_1 = Canva_1()
        self.cv_1_bt = NavigationToolbar2QT(self.cv_1, self)
        self.cv_1_bt.coordinates = False
        unwanted_buttons = ["Back", "Forward", "Customize"]
        for i in self.cv_1_bt.actions():  # setting navigation toolbar
            if i.text() in unwanted_buttons:
                self.cv_1_bt.removeAction(i)
        self.button_1 = QtWidgets.QPushButton("Generate Signal")
        self.button_2 = QtWidgets.QPushButton("Show")
        self.button_2.setEnabled(False)
        self.list_1 = QtWidgets.QListWidget(self)
        self.hl_1 = QtWidgets.QHBoxLayout()
        self.hl_1.addWidget(self.cv_1, 1)
        self.vl_1 = QtWidgets.QVBoxLayout()
        self.vl_1.addWidget(self.cv_1_bt)
        self.vl_1.addWidget(self.button_1)
        self.vl_1.addWidget(self.button_2)
        self.vl_1.addWidget(self.list_1)
        self.hl_1.addLayout(self.vl_1)
        self.gb_1.setLayout(self.hl_1)

        self.gb_2 = QtWidgets.QGroupBox("DFT", self)
        self.gb_2.setGeometry(1240, 50, 600, 300)
        self.cv_2 = Canva_2()
        self.hl_2 = QtWidgets.QHBoxLayout()
        self.hl_2.addWidget(self.cv_2)
        self.gb_2.setLayout(self.hl_2)

        self.gb_3 = QtWidgets.QGroupBox("STFT", self)
        self.gb_3.setGeometry(20, 360, 900, 500)
        self.cv_3 = Canva_3()
        self.hl_3 = QtWidgets.QHBoxLayout()
        self.hl_3.addWidget(self.cv_3)
        self.gb_3.setLayout(self.hl_3)

        self.gb_4 = QtWidgets.QGroupBox("CWT", self)
        self.gb_4.setGeometry(935, 360, 905, 500)
        self.cv_4 = Canva_4()
        self.hl_4 = QtWidgets.QHBoxLayout()
        self.hl_4.addWidget(self.cv_4)
        self.gb_4.setLayout(self.hl_4)

        # -- VALUE -- #
        self.n_data = 5000
        self.dat = [0] * self.n_data

        self.main_Program()

    def main_Program(self):
        self.menu_file_open.triggered.connect(lambda: self.open_file())
        self.menu_setting_DFT.triggered.connect(lambda: self.f_DFT())
        self.menu_setting_STFT.triggered.connect(lambda: self.f_STFT())
        self.menu_setting_CWT.triggered.connect(lambda: self.f_CWT())

        self.button_1.clicked.connect(lambda: self.signal_gen())
        self.button_2.clicked.connect(lambda: self.plot_1())

    def signal_gen(self):
        self.fs = 250  # 250 for 4 signal
        freq = [75, 50, 25, 10]
        for i in range(1250):
            self.dat[i] = np.sin(2 * np.pi * freq[0] * i / self.fs)
        for i in range(1251, 2500):
            self.dat[i] = np.sin(2 * np.pi * freq[1] * i / self.fs)
        for i in range(2501, 3750):
            self.dat[i] = np.sin(2 * np.pi * freq[2] * i / self.fs)
        for i in range(3751, 5000):
            self.dat[i] = np.sin(2 * np.pi * freq[3] * i / self.fs)

        self.list_1.clear()
        self.list_1.addItem("no.\tdata")
        for i in range(self.n_data):
            self.list_1.addItem(str(i) + "\t" + str("{:.3f}".format(self.dat[i])))

        self.button_2.setEnabled(True)

    def open_file(self):
        self.fs = 75  # for murmur.dat
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load File')
        self.dat = np.loadtxt(file_name)[:5000]

        self.list_1.clear()
        self.list_1.addItem("no.\tdata")
        for i in range(self.n_data):
            self.list_1.addItem(str(i) + "\t" + str("{:.3f}".format(self.dat[i])))

        self.button_2.setEnabled(True)

    def plot_1(self):
        self.cv_1.axes_1.clear()
        x_1 = np.arange(self.n_data)
        self.cv_1.axes_1.plot(x_1/self.fs, self.dat)
        self.cv_1.axes_1.set_xlim(0, self.n_data/self.fs)
        self.cv_1.axes_1.set_ylim(-80, 80)
        self.cv_1.draw()

        self.menu_setting_DFT.setEnabled(True)
        self.menu_setting_STFT.setEnabled(True)
        self.menu_setting_CWT.setEnabled(True)

    def f_DFT(self):
        n = np.arange(self.n_data)
        k = n.reshape((self.n_data, 1))
        # e = np.exp(-2j * np.pi * k * n / N)
        e = np.cos(2 * np.pi * k * n / self.n_data) - 1j * np.sin(2 * np.pi * k * n / self.n_data)
        y = np.dot(e, self.dat)  # dft

        freq = n * self.fs / self.n_data
        self.cv_2.axes_2.clear()
        self.cv_2.axes_2.stem(freq[:int(self.n_data / 2)], abs(y[:int(self.n_data / 2)]), 'b', markerfmt=" ",
                              basefmt="-b")
        self.cv_2.draw()

    def f_STFT(self):
        stft_window = 500
        f, t, y = signal.stft(self.dat, self.fs, nperseg=stft_window)
        self.cv_3.axes_3.pcolormesh((t * self.fs), f, np.abs(y), shading='gouraud')
        # self.cv_3.axes_3.pcolormesh((t * self.fs), f[:9], np.abs(y[:9, :]), shading='gouraud')
        self.cv_3.draw()

    def f_CWT(self):
        freq = np.arange(1, 120)  # set the target freq
        w0 = 2 * np.pi * 0.849
        scale = w0 * self.fs / (2 * freq * np.pi)

        n = np.arange(self.n_data)
        y = signal.cwt(self.dat, signal.morlet2, scale, dtype="complex128")

        self.cv_4.axes_4.pcolormesh(n, freq, np.abs(y), shading='gouraud')
        self.cv_4.draw()


class Canva_1(FigureCanvasQTAgg):
    def __init__(self):
        figure_1 = Figure(figsize=(9, 3), dpi=110)
        self.axes_1 = figure_1.add_subplot(111)
        super(Canva_1, self).__init__(figure_1)


class Canva_2(FigureCanvasQTAgg):
    def __init__(self):
        figure_2 = Figure(figsize=(9, 3), dpi=110)
        self.axes_2 = figure_2.add_subplot(111)
        super(Canva_2, self).__init__(figure_2)


class Canva_3(FigureCanvasQTAgg):
    def __init__(self):
        figure_3 = Figure(figsize=(9, 3), dpi=110)
        self.axes_3 = figure_3.add_subplot(111)
        super(Canva_3, self).__init__(figure_3)


class Canva_4(FigureCanvasQTAgg):
    def __init__(self):
        figure_4 = Figure(figsize=(9, 3), dpi=110)
        self.axes_4 = figure_4.add_subplot(111)
        super(Canva_4, self).__init__(figure_4)


app = QApplication(sys.argv)
win = MainWindow()
win.show()
sys.exit(app.exec_())
