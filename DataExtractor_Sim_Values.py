#!/usr/bin/env python3
import sys
import numpy as np
from cmath import *
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
# import scipy
from scipy import signal
from multiprocessing import Process, Value, Array
import threading
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSizePolicy, QWidget,\
    QPushButton, QHBoxLayout, QFileDialog, QLineEdit, QLabel, QComboBox
from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread,
                          QThreadPool, pyqtSignal)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# /\ Needed additional math functions
def next_pow2(a):
    m = log(a)/log(2)
    a = int(m.real)
    if a < m.real:
        return a+1
    else:
        return a
# \/


# /\ Global Constants
c = 2.99796*10**8
n0 = 1.00027+0j
n2 = n0
# theta = 60.23*pi/180
theta = 60.73*pi/180
# \/

initial_l = 0.0
nres = []
kres = []
freqs = []
time_h = []
p_h = []
s_h = []
freq_plot = []
pf_plot = []
sf_plot = []
pha_p = []
pha_s = []
middy = True
file_ES = 'ESF.txt'
file_EP = 'EPF.txt'
RefDat = 'N1Ref.txt'
saveSpace = 'Results.txt'
Threader = 8
Decimal = '.'
upLim = 2.0
botLim = 0.5
ResShow = False


# /\ Optical Functions
def theta_trans(n_out):   # /Generate transmitted angle\
    return asin((n0/n_out)*sin(theta))


def rp12(n_in, n_out):    # /Generate parallel reflection coefficient\
    theta_in = theta_trans(n_in)
    theta_uit = theta_trans(n_out)
    return (n_out*cos(theta_in)-n_in*cos(theta_uit))/(n_out*cos(theta_in)+n_in*cos(theta_uit))


def rs12(n_in, n_out):    # /Generate perpendicular reflection coefficient\
    theta_in = theta_trans(n_in)
    theta_uit = theta_trans(n_out)
    return (n_in*cos(theta_in)-n_out*cos(theta_uit))/(n_in*cos(theta_in)+n_out*cos(theta_uit))


def tp12(n_in, n_out):    # /Generate parallel reflection coefficient\
    theta_in = theta_trans(n_in)
    theta_uit = theta_trans(n_out)
    return (2*n_in*cos(theta_in))/(n_out*cos(theta_in)+n_in*cos(theta_uit))


def ts12(n_in, n_out):    # /Generate perpendicular reflection coefficient\
    theta_in = theta_trans(n_in)
    theta_uit = theta_trans(n_out)
    return (2*n_in*cos(theta_in))/(n_in*cos(theta_in)+n_out*cos(theta_uit))


# def correction():
#     fp = open("Correction.txt", "r")
#     f = []
#     a = []
#     p = []
#     for line in fp:
#         temp = line.split()
#         f.append(float(temp[0])*10**12)
#         a.append(float(temp[1]))
#         p.append(float(temp[2]))
#     fp.close()
#     fr = np.asarray(f)
#     am = np.asarray(a)
#     ph = np.asarray(p)
#     return fr, am, ph


def gauss(x, a=1, b=1, ccd=1):
    return a * np.exp(-(x-b)**2 / (2 * ccd**2))


def lulu(bb, arr):
    le = np.zeros(bb + 1)
    ue = np.zeros(bb + 1)

    # arrt = np.copy(arr)
    arrt = np.zeros(len(arr) + 2*bb)
    for lm in range(0, len(arr)):
        if lm < bb:
            arrt[lm] = arr[0]
        elif lm > len(arr) + bb:
            arrt[lm] = arr[-1]
        else:
            arrt[lm] = arr[lm - bb]

    for lm in range(bb, len(arrt)-bb):
        for lm2 in range(0, bb+1):
            aa = arrt[lm-bb + lm2: lm + lm2 + 1]
            le[lm2] = np.min(aa)
        arrt[lm] = np.max(le)

    for lm in range(bb, len(arrt)-bb):
        for lm2 in range(0, bb+1):
            aa = arrt[lm-bb + lm2: lm + lm2 + 1]
            ue[lm2] = np.max(aa)
        arrt[lm] = np.min(ue)
    nlu = np.copy(arrt)

    arrt = np.zeros(len(arr) + 2 * bb)
    for lm in range(0, len(arr)):
        if lm < bb:
            arrt[lm] = arr[0]
        elif lm > len(arr) + bb:
            arrt[lm] = arr[-1]
        else:
            arrt[lm] = arr[lm - bb]

    for lm in range(bb, len(arrt) - bb):
        for lm2 in range(0, bb + 1):
            aa = arrt[lm - bb + lm2: lm + lm2 + 1]
            ue[lm2] = np.max(aa)
        arrt[lm] = np.min(ue)

    for lm in range(bb, len(arrt) - bb):
        for lm2 in range(0, bb + 1):
            aa = arrt[lm - bb + lm2: lm + lm2 + 1]
            le[lm2] = np.min(aa)
        arrt[lm] = np.max(le)

    nul = np.copy(arrt)

    for lm in range(bb, len(arrt) - bb):
        if nlu[lm] == nul[lm]:
            arr[lm - bb] = nlu[lm]
        elif np.abs(arr[lm - bb] - nlu[lm]) > np.abs(arr[lm - bb] - nul[lm]):
            arr[lm - bb] = nlu[lm]
        else:
            arr[lm - bb] = nul[lm]

    return arr

# \/


class Worker1:
    def __init__(self):
        self.WorkerNumber = 1
        self.TotalWorkers = 1
        self.initial_l1 = 0.0003
        self.l1 = self.initial_l1 / cos(theta)
        self.current1 = 0
        self.initSet1 = np.zeros(2)
        self.Counting = 0
        self.StartZone = 0
        self.Zone = 0
        self.StartZone2 = 0
        self.time = np.zeros(2)
        self.FreqS1 = np.zeros(2)
        self.YPP = np.zeros(2)
        self.YSS = np.zeros(2)
        self.h_experiment = np.zeros(2, dtype=complex)
        self.nA1 = np.zeros(2, dtype=complex)
        self.EXAngS1 = np.zeros(2)
        self.EXMagS1 = np.zeros(2)
        self.ln = 0
        self.freq = np.zeros(2)
        self.y_s_freq = np.zeros(2, dtype=complex)
        self.y_p_freq = np.zeros(2, dtype=complex)
        self.y_s_freqi = np.zeros(2, dtype=complex)
        self.y_p_freqi = np.zeros(2, dtype=complex)
        self.topLim = 4.0
        self.botLim = 0.3
        self.decimal = '.'
        self.middy = False

    def set_thread(self, num, tot):
        self.WorkerNumber = num
        self.TotalWorkers = tot

    def set_lims(self, bot, top):
        self.botLim = bot
        self.topLim = top

    def set_dec(self, dec):
        self.decimal = dec

    def set_mid(self, mid):
        self.middy = mid

    def set_files(self, es, ep):
        yp = open(ep, 'r')
        ys = open(es, 'r')

        xt = []
        ypt = []
        yst = []
        if self.decimal == '.':
            for line in yp:
                temp2 = line.split('\t')
                if not temp2[0]:
                    break
                xt.append(float(temp2[0]) * 10 ** -12)
                ypt.append(complex(temp2[1]))
            yp.close()

            for line in ys:
                temp2 = line.split('\t')
                if not temp2[0]:
                    break
                yst.append(complex(temp2[1]))
            ys.close()

        else:
            for line in yp:
                temp2 = line.replace(self.decimal, '.').split('\t')
                if not temp2[0]:
                    break
                xt.append(float(temp2[0]) * 10 ** -12)
                ypt.append(complex(temp2[1]))
            yp.close()

            for line in ys:
                temp2 = line.replace(self.decimal, '.').split('\t')
                if not temp2[0]:
                    break
                yst.append(complex(temp2[1]))
            ys.close()

        n_fft = 2 ** (next_pow2(len(xt)))

        self.time = np.asarray(xt)
        self.YPP = np.asarray(ypt)
        self.YSS = np.asarray(yst)

        start2 = 0
        # for i in range(0, len(self.time)):
        #     if self.time[i] >= 9.17 * 10 ** -12:
        #         start2 = i
        #         break
        #
        # stop2 = 0
        # for i in range(start2, len(self.time)):
        #     if self.time[i] > 12.59 * 10 ** -12:
        #         stop2 = i
        #         break
        for i in range(0, len(self.time)):
            if self.time[i] >= -1.6 * 10 ** -12:
                start2 = i
                break

        stop2 = 0
        for i in range(start2, len(self.time)):
            if self.time[i] > 2.0 * 10 ** -12:
                stop2 = i
                break
        bread = (stop2 - start2) / (2 * sqrt(2 * np.log(2)))
        gg = gauss(np.linspace(0, len(self.time), len(self.time)), b=np.argmax(np.abs(self.YSS)), ccd=bread)

        self.freq = (1 / abs(self.time[1] - self.time[0])) * (np.arange(n_fft) - n_fft / 2) / n_fft

        if self.middy:
            self.y_p_freq = np.fft.fftshift(np.fft.fft(self.YPP, n_fft))
            self.y_s_freq = np.fft.fftshift(np.fft.fft(self.YSS, n_fft))
        else:
            self.y_p_freq = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.YPP), n_fft))
            self.y_s_freq = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.YSS), n_fft))

        if self.middy:
            self.y_p_freqi = np.fft.fftshift(np.fft.fft(self.YPP*gg, n_fft))
            self.y_s_freqi = np.fft.fftshift(np.fft.fft(self.YSS*gg, n_fft))
        else:
            self.y_p_freqi = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.YPP*gg), n_fft))
            self.y_s_freqi = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.YSS*gg), n_fft))

        self.h_experiment = np.zeros(n_fft, dtype=complex)

        for i in range(0, n_fft):
            self.h_experiment[i] = self.y_p_freq[i]/self.y_s_freq[i]

        for i in range(0, n_fft):
            if self.freq[i] == 0:
                self.StartZone = i
                break

        for i in range(self.StartZone, n_fft):
            self.Zone += 1
            if self.freq[i] >= self.topLim * 10 ** 12:
                break

        endz = self.StartZone + self.Zone
        for i in range(self.StartZone, endz):
            if self.freq[i] >= self.botLim * 10 ** 12:
                self.StartZone2 = i
                break

        endz = self.StartZone + self.Zone
        cuts = endz - self.StartZone2
        checkety = cuts % self.TotalWorkers

        if checkety != 0:
            cuts = cuts + self.TotalWorkers - checkety

        t_end = self.StartZone2 + cuts
        freq_s = self.freq[self.StartZone2:t_end]
        exh = self.h_experiment[self.StartZone2:t_end]
        self.ln = int(len(exh) / self.TotalWorkers)
        exh1 = exh[self.WorkerNumber*self.ln: (self.WorkerNumber*self.ln + self.ln)]
        self.FreqS1 = freq_s[self.WorkerNumber*self.ln: (self.WorkerNumber*self.ln + self.ln)]
        self.nA1 = np.zeros(self.ln, dtype=complex)

        self.EXAngS1 = np.unwrap(np.angle(exh1))
        self.EXMagS1 = np.abs(exh1)

    def get_freqs(self):
        return self.FreqS1

    def get_ln(self):
        return self.ln

    def get_orig(self):
        return self.time, self.YPP, self.YSS, self.freq, self.y_p_freq, self.y_s_freq, self.y_p_freqi, self.y_s_freqi

    def set_len(self, leng):
        self.initial_l1 = leng
        self.l1 = self.initial_l1 / cos(theta)


class Worker2:
    def __init__(self):
        self.initial_l1 = 0.0003
        self.l1 = self.initial_l1 / cos(theta)
        self.current1 = 0
        self.initSet1 = np.zeros(2)
        self.Counting = 0
        self.FreqS1 = np.zeros(2)
        self.YPP = np.zeros(2)
        self.YSS = np.zeros(2)
        self.nA1 = np.zeros(2, dtype=complex)
        self.nA1t = np.zeros(2, dtype=complex)
        self.EXAngS1 = np.zeros(2)
        self.EXMagS1 = np.zeros(2)
        self.EXAngS2 = np.zeros(2)
        self.EXMagS2 = np.zeros(2)
        self.EXAngSi = np.zeros(2)
        self.EXMagSi = np.zeros(2)
        self.ln = 0
        self.freq = np.zeros(2)
        self.y_s_freq = np.zeros(2, dtype=complex)
        self.y_p_freq = np.zeros(2, dtype=complex)
        self.y_s_freqi = np.zeros(2, dtype=complex)
        self.y_p_freqi = np.zeros(2, dtype=complex)
        self.es = np.zeros(2, dtype=complex)
        self.ep = np.zeros(2, dtype=complex)
        self.esi = np.zeros(2, dtype=complex)
        self.epi = np.zeros(2, dtype=complex)

    def set_input(self, fr, es, ep, esi, epi):
        self.FreqS1 = np.zeros(len(fr))
        self.es = np.zeros(len(fr), dtype=complex)
        self.ep = np.zeros(len(fr), dtype=complex)
        self.esi = np.zeros(len(fr), dtype=complex)
        self.epi = np.zeros(len(fr), dtype=complex)
        exh = np.zeros(len(fr), dtype=complex)
        for wr in range(0, len(fr)):
            self.FreqS1[wr] = fr[wr]
            self.es[wr] = es[wr]
            self.ep[wr] = ep[wr]
            self.esi[wr] = esi[wr]
            self.epi[wr] = epi[wr]
            exh[wr] = ep[wr]/es[wr]
        self.ln = int(len(exh))
        self.nA1 = np.zeros(self.ln, dtype=complex)

        self.EXAngS1 = np.unwrap(np.angle(ep))-np.unwrap(np.angle(es))
        self.EXAngS2 = np.unwrap(np.angle(es)) - np.unwrap(np.angle(ep))
        self.EXMagS1 = np.abs(ep)/np.abs(es)
        self.EXMagS2 = np.abs(es) / np.abs(ep)
        self.EXAngSi = np.unwrap(np.angle(epi)) - np.unwrap(np.angle(esi))
        self.EXMagSi = np.abs(epi) / np.abs(esi)

    def initial_generator1(self, mag, pha):
        for x in range(0, self.ln):
            p = mag[x] * (e ** (1j * (-pha[x])))
            eps = (n0 ** 2) * ((sin(theta)) ** 2) * (1 + (((1 - p) / (1 + p)) ** 2) * (tan(theta)) ** 2)
            a = eps.real
            b = eps.imag
            self.nA1[x] = (sqrt((a + sqrt(a ** 2 + (b ** 2))) / 2)).real
            self.nA1[x] -= 1j * (b / (2 * self.nA1[x]))

    def attenuation1(self, n, f):  # /Returns attenuation factor for a given complex refractive index and frequency\
        s0 = sqrt(1 - ((n0 * sin(theta)) / n) ** 2)
        self.l1 = self.initial_l1 / s0
        return e ** (-1j * 2 * pi * f * n * self.l1 / c)  # TODO: Double Check if 4 should be 2

    def h_theory1p(self, f, n1):  # /Returns the theoretical Transfer function ratio for a given frequency and complex
                                    # refractive index\
        p1 = self.attenuation1(n1, f) ** 2

        rp0 = rp12(n0, n1)

        rp1 = rp12(n1, n0)

        rp2 = rp12(n1, n2)

        tp0 = (n0/n1)*(rp0 + 1)

        tp1 = (n1 / n0) * (rp1 + 1)

        ap1 = p1 * tp0 * tp1 * rp2

        ap2 = 1 - p1 * rp1 * rp2

        h2 = rp0 + ap1/ap2

        return h2

    def h_theory1s(self, f, n1):  # /Returns the theoretical Transfer function ratio for a given frequency and complex
                                    # refractive index\
        p1 = self.attenuation1(n1, f) ** 2

        rs0 = rs12(n0, n1)

        rs1 = rs12(n1, n0)

        rs2 = rs12(n1, n2)

        ts0 = rs0 + 1

        ts1 = rs1 + 1

        as1 = p1 * ts0 * ts1 * rs2

        as2 = 1 - p1 * rs1 * rs2

        h1 = rs0 + as1/as2

        return h1

    def err1(self, n1):    # Returns the error between the theoretical transfer function ratio and extracted transfer
                            # function ratio after changing the complex refractive index at the currently indicated
                            # index to a given complex refractive index
        self.nA1t = np.copy(self.nA1)
        self.nA1t[self.current1] = n1[0]+1j*n1[1]

        htp = np.zeros(self.ln, dtype=complex)
        hts = np.zeros(self.ln, dtype=complex)
        for q in range(0, self.current1 + 1):
            htap = self.h_theory1p(self.FreqS1[q], self.nA1t[q])
            htas = self.h_theory1s(self.FreqS1[q], self.nA1t[q])
            hta_r = float(htap.real)  # /
            hta_i = float(htap.imag)  # numpy problem workaround
            htp[q] = hta_r + 1j * hta_i  # \
            hta_r = float(htas.real)  # /
            hta_i = float(htas.imag)  # numpy problem workaround
            hts[q] = hta_r + 1j * hta_i  # \

        htm = np.abs(htp[self.current1]) / np.abs(hts[self.current1])
        htp2 = np.unwrap(np.angle(htp))[self.current1] - np.unwrap(np.angle(hts))[self.current1]

        mm = self.EXMagS1[self.current1] - htm
        aa = self.EXAngS1[self.current1] - htp2

        er = np.abs(mm) + np.abs(aa)
        return er

    def err2(self, n1):    # Returns the error between the theoretical transfer function ratio and extracted transfer
                            # function ratio after changing the complex refractive index at the currently indicated
                            # index to a given complex refractive index
        self.nA1t = np.copy(self.nA1)
        self.nA1t[self.current1] = n1[0]+1j*n1[1]

        htp = np.zeros(self.ln, dtype=complex)
        hts = np.zeros(self.ln, dtype=complex)
        for q in range(0, self.current1 + 1):
            htap = self.h_theory1p(self.FreqS1[q], self.nA1t[q])
            htas = self.h_theory1s(self.FreqS1[q], self.nA1t[q])
            hta_r = float(htap.real)  # /
            hta_i = float(htap.imag)  # numpy problem workaround
            htp[q] = hta_r + 1j * hta_i  # \
            hta_r = float(htas.real)  # /
            hta_i = float(htas.imag)  # numpy problem workaround
            hts[q] = hta_r + 1j * hta_i  # \

        htm = np.abs(hts[self.current1]) / np.abs(htp[self.current1])
        htp2 = np.unwrap(np.angle(hts))[self.current1] - np.unwrap(np.angle(htp))[self.current1]

        mm = self.EXMagS2[self.current1] - htm
        aa = self.EXAngS2[self.current1] - htp2

        er = np.abs(mm) + np.abs(aa)
        return er

    def err3(self, n1):    # Returns the error between the theoretical transfer function ratio and extracted transfer
                            # function ratio after changing the complex refractive index at the currently indicated
                            # index to a given complex refractive index
        self.nA1t = np.copy(self.nA1)
        self.nA1t[self.current1] = n1[0]+1j*n1[1]

        htp = np.zeros(self.ln, dtype=complex)
        hts = np.zeros(self.ln, dtype=complex)
        for q in range(0, self.current1 + 1):
            htap = self.h_theory1p(self.FreqS1[q], self.nA1t[q])
            htas = self.h_theory1s(self.FreqS1[q], self.nA1t[q])
            hta_r = float(htap.real)  # /
            hta_i = float(htap.imag)  # numpy problem workaround
            htp[q] = hta_r + 1j * hta_i  # \
            hta_r = float(htas.real)  # /
            hta_i = float(htas.imag)  # numpy problem workaround
            hts[q] = hta_r + 1j * hta_i  # \

        htm = np.abs(hts[self.current1])
        htp2 = np.unwrap(np.angle(hts))[self.current1]

        mm = np.abs(self.es)[self.current1] - htm
        aa = np.unwrap(np.angle(self.es))[self.current1] - htp2

        er = np.abs(mm) + np.abs(aa)
        return er

    def smoothness1(self, thick):  # Returns the smoothness of the complex refractive functions after changing the
                                    # sample thickness to a given quantity
        nmin = Array('d', range(len(self.FreqS1)))
        kmin = Array('d', range(len(self.FreqS1)))
        lt = Value('d', thick)
        recur2(self.FreqS1, self.es, self.ep, self.esi, self.epi, nmin, kmin, lt)

        bb = 5

        nkep = np.zeros(len(self.FreqS1))
        kkep = np.zeros(len(self.FreqS1))
        ksi = int(np.log10(np.abs(kmin[1])))
        for lm in range(0, len(self.FreqS1)):
            nkep[lm] = round(nmin[lm], 3)
            kkep[lm] = round(-kmin[lm]/(10**ksi), 3)*(10**ksi)

        nkep = lulu(bb, nkep)
        kkep = lulu(bb, kkep)

        for lm in range(bb, len(self.FreqS1) - bb):
            nmin[lm] = nkep[lm]
            kmin[lm] = -kkep[lm]

        #
        # recur3(self.FreqS1, self.es, self.ep, nmin, kmin, lt)
        # #
        # bb = 2
        #
        # nkep = np.zeros(len(self.FreqS1))
        # kkep = np.zeros(len(self.FreqS1))
        # for lm in range(0, len(self.FreqS1)):
        #     nkep[lm] = round(nmin[lm], 3)
        #     kkep[lm] = round(-kmin[lm]/(10**ksi), 3)*(10**ksi)
        #
        # nkep = lulu(bb, nkep)
        # kkep = lulu(bb, kkep)
        #
        # for lm in range(bb, len(self.FreqS1) - bb):
        #     nmin[lm] = nkep[lm]
        #     kmin[lm] = -kkep[lm]

        # recur3(self.FreqS1, self.es, self.ep, nmin, kmin, lt)

        self.nA1 = np.zeros(len(self.FreqS1), dtype=complex)
        for zzk in range(0, len(self.FreqS1)):
            self.nA1[zzk] = nmin[zzk] + 1j*abs(kmin[zzk])
        tt = 0
        for m in range(3, self.ln-3):  # Skip first element due to unstable nature
            de = np.abs(float(self.nA1[m-1].real)-float(self.nA1[m].real)) \
                 + np.abs(float(self.nA1[m-1].imag)-float(self.nA1[m].imag))
            tt += de
        return tt

    def call1(self):
        # /\ Use minimisation function to find the thickness of the sample
        shift = np.log10(self.initial_l1)
        if shift < int(shift):
            shift = 10**(int(shift)-2)
        else:
            shift = 10**int(shift-1)
        res = minimize_scalar(self.smoothness1, bounds=(self.initial_l1-shift, self.initial_l1+shift),
                              method='bounded', options={'xatol': 1e-8})

        self.initial_l1 = float(res.x)
    # \/
        return self.initial_l1

    def call2(self):
        # /\ Use minimisation function to extract the complex refractive index of the sample
        self.initial_generator1(self.EXMagSi, self.EXAngSi)
        # self.initial_generator1(self.EXMagS1, self.EXAngS1)
        # ka = 0
        # na = 0
        # for i in range(0, self.ln):
        #     na += self.nA1[i].real / self.ln
        #     ka += self.nA1[i].imag / self.ln
        # self.nA1 = (na + 1j * ka) * np.ones(self.ln, dtype=complex)

        self.current1 = 0

        for zk in range(0, self.ln):
            self.initSet1[0] = self.nA1[zk].real
            self.initSet1[1] = self.nA1[zk].imag
            res = minimize(self.err1, self.initSet1, method='nelder-mead',
                           options={'xtol': 1e-8, 'maxiter': 1000, 'disp': False})
            self.nA1[zk] = res.x[0] + 1j*res.x[1]
            self.current1 += 1

        return self.nA1

    def get_ln(self):
        return self.ln

    def set_len(self, leng):
        self.initial_l1 = leng
        self.l1 = self.initial_l1 / cos(theta)


# class Worker3:
#     def __init__(self):
#         self.initial_l1 = 0.0003
#         self.l1 = self.initial_l1 / cos(theta)
#         self.current1 = 0
#         self.initSet1 = np.zeros(2)
#         self.Counting = 0
#         self.FreqS1 = np.zeros(2)
#         self.YPP = np.zeros(2)
#         self.YSS = np.zeros(2)
#         self.nA1 = np.zeros(2, dtype=complex)
#         self.nA1t = np.zeros(2, dtype=complex)
#         self.EXAngS1 = np.zeros(2)
#         self.EXMagS1 = np.zeros(2)
#         self.EXAngS2 = np.zeros(2)
#         self.EXMagS2 = np.zeros(2)
#         self.ln = 0
#         self.freq = np.zeros(2)
#         self.y_s_freq = np.zeros(2, dtype=complex)
#         self.y_p_freq = np.zeros(2, dtype=complex)
#         self.es = np.zeros(2, dtype=complex)
#         self.ep = np.zeros(2, dtype=complex)
#         self.esi = np.zeros(2, dtype=complex)
#         self.epi = np.zeros(2, dtype=complex)
#
#     def set_input(self, fr, es, ep, nr, kr):
#         self.FreqS1 = np.zeros(len(fr))
#         self.es = np.zeros(len(fr), dtype=complex)
#         self.ep = np.zeros(len(fr), dtype=complex)
#         exh = np.zeros(len(fr), dtype=complex)
#         self.nA1 = np.zeros(len(fr), dtype=complex)
#         for wr in range(0, len(fr)):
#             self.FreqS1[wr] = fr[wr]
#             self.es[wr] = es[wr]
#             self.ep[wr] = ep[wr]
#             self.nA1[wr] = nr[wr] + 1j*kr[wr]
#             exh[wr] = ep[wr] / es[wr]
#         self.ln = int(len(exh))
#
#         self.EXAngS1 = np.unwrap(np.angle(ep)) - np.unwrap(np.angle(es))
#         self.EXAngS2 = np.unwrap(np.angle(es)) - np.unwrap(np.angle(ep))
#         self.EXMagS1 = np.abs(ep) / np.abs(es)
#         self.EXMagS2 = np.abs(es) / np.abs(ep)
#
#     def attenuation1(self, n, f):  # /Returns attenuation factor for a given complex refractive index and frequency\
#         s0 = sqrt(1 - ((n0 * sin(theta)) / n) ** 2)
#         self.l1 = self.initial_l1 / s0
#         return e ** (-1j * 2 * pi * f * n * self.l1 / c)  # TODO: Double Check if 4 should be 2
#
#     def h_theory1p(self, f,
#                    n1):  # /Returns the theoretical Transfer function ratio for a given frequency and complex
#         # refractive index\
#         p1 = self.attenuation1(n1, f) ** 2
#
#         rp0 = rp12(n0, n1)
#
#         rp1 = rp12(n1, n0)
#
#         rp2 = rp12(n1, n2)
#
#         tp0 = (n0 / n1) * (rp0 + 1)
#
#         ap1 = p1 * tp0 * tp0 * rp2
#
#         ap2 = 1 - p1 * rp1 * rp2
#
#         h2 = rp0 + ap1 / ap2
#
#         return h2
#
#     def h_theory1s(self, f,
#                    n1):  # /Returns the theoretical Transfer function ratio for a given frequency and complex
#         # refractive index\
#         p1 = self.attenuation1(n1, f) ** 2
#
#         rs0 = rs12(n0, n1)
#
#         rs1 = rs12(n1, n0)
#
#         rs2 = rs12(n1, n2)
#
#         ts0 = rs0 + 1
#
#         as1 = p1 * ts0 * ts0 * rs2
#
#         as2 = 1 - p1 * rs1 * rs2
#
#         h1 = rs0 + as1 / as2
#
#         return h1
#
#     def err1(self, n1):  # Returns the error between the theoretical transfer function ratio and extracted transfer
#         # function ratio after changing the complex refractive index at the currently indicated
#         # index to a given complex refractive index
#         self.nA1t = np.copy(self.nA1)
#         self.nA1t[self.current1] = n1[0] + 1j * n1[1]
#
#         htp = np.zeros(self.ln, dtype=complex)
#         hts = np.zeros(self.ln, dtype=complex)
#         for q in range(0, self.current1 + 1):
#             htap = self.h_theory1p(self.FreqS1[q], self.nA1t[q])
#             htas = self.h_theory1s(self.FreqS1[q], self.nA1t[q])
#             hta_r = float(htap.real)  # /
#             hta_i = float(htap.imag)  # numpy problem workaround
#             htp[q] = hta_r + 1j * hta_i  # \
#             hta_r = float(htas.real)  # /
#             hta_i = float(htas.imag)  # numpy problem workaround
#             hts[q] = hta_r + 1j * hta_i  # \
#
#         htm = np.abs(htp[self.current1]) / np.abs(hts[self.current1])
#         htp2 = np.unwrap(np.angle(htp))[self.current1] - np.unwrap(np.angle(hts))[self.current1]
#
#         mm = self.EXMagS1[self.current1] - htm
#         aa = self.EXAngS1[self.current1] - htp2
#
#         er = np.abs(mm) + np.abs(aa)
#         return er
#
#     def err2(self, n1):  # Returns the error between the theoretical transfer function ratio and extracted transfer
#         # function ratio after changing the complex refractive index at the currently indicated
#         # index to a given complex refractive index
#         self.nA1t = np.copy(self.nA1)
#         self.nA1t[self.current1] = n1[0] + 1j * n1[1]
#
#         htp = np.zeros(self.ln, dtype=complex)
#         hts = np.zeros(self.ln, dtype=complex)
#         for q in range(0, self.current1 + 1):
#             htap = self.h_theory1p(self.FreqS1[q], self.nA1t[q])
#             htas = self.h_theory1s(self.FreqS1[q], self.nA1t[q])
#             hta_r = float(htap.real)  # /
#             hta_i = float(htap.imag)  # numpy problem workaround
#             htp[q] = hta_r + 1j * hta_i  # \
#             hta_r = float(htas.real)  # /
#             hta_i = float(htas.imag)  # numpy problem workaround
#             hts[q] = hta_r + 1j * hta_i  # \
#
#         htm = np.abs(hts[self.current1]) / np.abs(htp[self.current1])
#         htp2 = np.unwrap(np.angle(hts))[self.current1] - np.unwrap(np.angle(htp))[self.current1]
#
#         mm = self.EXMagS2[self.current1] - htm
#         aa = self.EXAngS2[self.current1] - htp2
#
#         er = np.abs(mm) + np.abs(aa)
#         return er
#
#     def call2(self):
#         # /\ Use minimisation function to extract the complex refractive index of the sample
#
#         self.current1 = 0
#
#         for i in range(0, self.ln):
#             self.initSet1[0] = self.nA1[i].real
#             self.initSet1[1] = self.nA1[i].imag
#             res = minimize(self.err2, self.initSet1, method='nelder-mead',
#                            options={'xtol': 1e-8, 'maxiter': 1000, 'disp': False})
#             self.nA1[i] = res.x[0] + 1j * res.x[1]
#             self.current1 += 1
#         # print('hello')
#         return self.nA1
#
#     def get_ln(self):
#         return self.ln
#
#     def set_len(self, leng):
#         self.initial_l1 = leng
#         self.l1 = self.initial_l1 / cos(theta)
class Worker1k:
    def __init__(self):
        self.WorkerNumber = 1
        self.TotalWorkers = 1
        self.initial_l1 = 0.0003
        self.l1 = self.initial_l1 / cos(theta)
        self.current1 = 0
        self.initSet1 = np.zeros(2)
        self.Counting = 0
        self.StartZone = 0
        self.Zone = 0
        self.StartZone2 = 0
        self.time = np.zeros(2)
        self.FreqS1 = np.zeros(2)
        self.YPP = np.zeros(2)
        self.YSS = np.zeros(2)
        self.h_experiment = np.zeros(2, dtype=complex)
        self.nA1 = np.zeros(2, dtype=complex)
        self.EXAngS1 = np.zeros(2)
        self.EXMagS1 = np.zeros(2)
        self.ln = 0
        self.freq = np.zeros(2)
        self.y_s_freq = np.zeros(2, dtype=complex)
        self.y_p_freq = np.zeros(2, dtype=complex)
        self.y_s_freqi = np.zeros(2, dtype=complex)
        self.y_p_freqi = np.zeros(2, dtype=complex)
        self.y_s_freqi2 = np.zeros(2, dtype=complex)
        self.y_p_freqi2 = np.zeros(2, dtype=complex)
        self.topLim = 4.0
        self.botLim = 0.3
        self.decimal = '.'
        self.middy = False

    def set_thread(self, num, tot):
        self.WorkerNumber = num
        self.TotalWorkers = tot

    def set_lims(self, bot, top):
        self.botLim = bot
        self.topLim = top

    def set_dec(self, dec):
        self.decimal = dec

    def set_mid(self, mid):
        self.middy = mid

    def set_files(self, es, ep):
        yp = open(ep, 'r')
        ys = open(es, 'r')

        xt = []
        ypt = []
        yst = []
        if self.decimal == '.':
            for line in yp:
                temp2 = line.split('\t')
                if not temp2[0]:
                    break
                xt.append(float(temp2[0]) * 10 ** -12)
                ypt.append(complex(temp2[1]))
            yp.close()

            for line in ys:
                temp2 = line.split('\t')
                if not temp2[0]:
                    break
                yst.append(complex(temp2[1]))
            ys.close()

        else:
            for line in yp:
                temp2 = line.replace(self.decimal, '.').split('\t')
                if not temp2[0]:
                    break
                xt.append(float(temp2[0]) * 10 ** -12)
                ypt.append(complex(temp2[1]))
            yp.close()

            for line in ys:
                temp2 = line.replace(self.decimal, '.').split('\t')
                if not temp2[0]:
                    break
                yst.append(complex(temp2[1]))
            ys.close()

        n_fft = 2 ** (next_pow2(len(xt)))

        self.time = np.asarray(xt)
        self.YPP = np.asarray(ypt)
        self.YSS = np.asarray(yst)

        start2 = 0
        for i in range(0, len(self.time)):
            if self.time[i] >= 9.1 * 10 ** -12:
                start2 = i
                break

        stop2 = 0
        for i in range(start2, len(self.time)):
            if self.time[i] > 12.5 * 10 ** -12:
                stop2 = i
                break

        start3 = 0
        for i in range(0, len(self.time)):
            if self.time[i] >= 19 * 10 ** -12:
                start3 = i
                break

        stop3 = 0
        for i in range(start2, len(self.time)):
            if self.time[i] > 22.6 * 10 ** -12:
                stop3 = i
                break

        bread = (stop2 - start2) / (2 * sqrt(2 * np.log(2)))
        bread2 = (stop3 - start3) / (2 * sqrt(2 * np.log(2)))
        gg = gauss(np.linspace(0, len(self.time), len(self.time)), b=np.argmax(np.abs(self.YSS)), ccd=bread)
        gg2 = gauss(np.linspace(0, len(self.time), len(self.time)),
                    b=np.argmax(np.abs(self.YSS - self.YSS * gg)), ccd=bread2)

        self.freq = (1 / abs(self.time[1] - self.time[0])) * (np.arange(n_fft) - n_fft / 2) / n_fft

        if self.middy:
            self.y_p_freq = np.fft.fftshift(np.fft.fft(self.YPP, n_fft))
            self.y_s_freq = np.fft.fftshift(np.fft.fft(self.YSS, n_fft))
        else:
            self.y_p_freq = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.YPP), n_fft))
            self.y_s_freq = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.YSS), n_fft))

        if self.middy:
            self.y_p_freqi = np.fft.fftshift(np.fft.fft(self.YPP*gg, n_fft))
            self.y_s_freqi = np.fft.fftshift(np.fft.fft(self.YSS*gg, n_fft))
        else:
            self.y_p_freqi = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.YPP*gg), n_fft))
            self.y_s_freqi = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.YSS*gg), n_fft))

        if self.middy:
            self.y_p_freqi2 = np.fft.fftshift(np.fft.fft(self.YPP*gg2, n_fft))
            self.y_s_freqi2 = np.fft.fftshift(np.fft.fft(self.YSS*gg2, n_fft))
        else:
            self.y_p_freqi2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.YPP*gg2), n_fft))
            self.y_s_freqi2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.YSS*gg2), n_fft))

        self.h_experiment = np.zeros(n_fft, dtype=complex)

        for i in range(0, n_fft):
            self.h_experiment[i] = self.y_p_freq[i]/self.y_s_freq[i]

        for i in range(0, n_fft):
            if self.freq[i] == 0:
                self.StartZone = i
                break

        for i in range(self.StartZone, n_fft):
            self.Zone += 1
            if self.freq[i] >= self.topLim * 10 ** 12:
                break

        endz = self.StartZone + self.Zone
        for i in range(self.StartZone, endz):
            if self.freq[i] >= self.botLim * 10 ** 12:
                self.StartZone2 = i
                break

        endz = self.StartZone + self.Zone
        cuts = endz - self.StartZone2
        checkety = cuts % self.TotalWorkers

        if checkety != 0:
            cuts = cuts + self.TotalWorkers - checkety

        t_end = self.StartZone2 + cuts
        freq_s = self.freq[self.StartZone2:t_end]
        exh = self.h_experiment[self.StartZone2:t_end]
        self.ln = int(len(exh) / self.TotalWorkers)
        exh1 = exh[self.WorkerNumber*self.ln: (self.WorkerNumber*self.ln + self.ln)]
        self.FreqS1 = freq_s[self.WorkerNumber*self.ln: (self.WorkerNumber*self.ln + self.ln)]
        self.nA1 = np.zeros(self.ln, dtype=complex)

        self.EXAngS1 = np.unwrap(np.angle(exh1))
        self.EXMagS1 = np.abs(exh1)

    def get_freqs(self):
        return self.FreqS1

    def get_ln(self):
        return self.ln

    def get_orig(self):
        return self.time, self.YPP, self.YSS, self.freq, self.y_p_freq, self.y_s_freq, self.y_p_freqi, self.y_s_freqi, \
               self.y_p_freqi2, self.y_s_freqi2

    def set_len(self, leng):
        self.initial_l1 = leng
        self.l1 = self.initial_l1 / cos(theta)


# TODO: worker k 2 needs lvl 1
class Worker2k:
    def __init__(self):
        self.initial_l1 = 0.0003
        self.l1 = self.initial_l1 / cos(theta)
        self.current1 = 0
        self.initSet1 = np.zeros(2)
        self.Counting = 0
        self.FreqS1 = np.zeros(2)
        self.YPP = np.zeros(2)
        self.YSS = np.zeros(2)
        self.nA1 = np.zeros(2, dtype=complex)
        self.nA2 = np.zeros(2, dtype=complex)
        self.nA2t = np.zeros(2, dtype=complex)
        self.EXAngS1 = np.zeros(2)
        self.EXMagS1 = np.zeros(2)
        self.EXAngS2 = np.zeros(2)
        self.EXMagS2 = np.zeros(2)
        self.EXAngSi = np.zeros(2)
        self.EXMagSi = np.zeros(2)
        self.EXAngSi2 = np.zeros(2)
        self.EXMagSi2 = np.zeros(2)
        self.ln = 0
        self.freq = np.zeros(2)
        self.y_s_freq = np.zeros(2, dtype=complex)
        self.y_p_freq = np.zeros(2, dtype=complex)
        self.y_s_freqi = np.zeros(2, dtype=complex)
        self.y_p_freqi = np.zeros(2, dtype=complex)
        self.y_s_freqi2 = np.zeros(2, dtype=complex)
        self.y_p_freqi2 = np.zeros(2, dtype=complex)
        self.es = np.zeros(2, dtype=complex)
        self.ep = np.zeros(2, dtype=complex)
        self.esi = np.zeros(2, dtype=complex)
        self.epi = np.zeros(2, dtype=complex)
        self.esi2 = np.zeros(2, dtype=complex)
        self.epi2 = np.zeros(2, dtype=complex)

    def set_input(self, fr, es, ep, esi, epi, esi2, epi2):
        self.FreqS1 = np.zeros(len(fr))
        self.es = np.zeros(len(fr), dtype=complex)
        self.ep = np.zeros(len(fr), dtype=complex)
        self.esi = np.zeros(len(fr), dtype=complex)
        self.epi = np.zeros(len(fr), dtype=complex)
        self.esi2 = np.zeros(len(fr), dtype=complex)
        self.epi2 = np.zeros(len(fr), dtype=complex)
        exh = np.zeros(len(fr), dtype=complex)
        for wr in range(0, len(fr)):
            self.FreqS1[wr] = fr[wr]
            self.es[wr] = es[wr]
            self.ep[wr] = ep[wr]
            self.esi[wr] = esi[wr]
            self.epi[wr] = epi[wr]
            self.esi2[wr] = esi2[wr]
            self.epi2[wr] = epi2[wr]
            exh[wr] = ep[wr] / es[wr]
        self.ln = int(len(exh))
        self.nA1 = np.zeros(self.ln, dtype=complex)
        self.nA2 = np.zeros(self.ln, dtype=complex)

        self.EXAngS1 = np.unwrap(np.angle(ep)) - np.unwrap(np.angle(es))
        self.EXAngS2 = np.unwrap(np.angle(es)) - np.unwrap(np.angle(ep))
        self.EXMagS1 = np.abs(ep) / np.abs(es)
        self.EXMagS2 = np.abs(es) / np.abs(ep)
        self.EXAngSi = np.unwrap(np.angle(epi)) - np.unwrap(np.angle(esi))
        self.EXMagSi = np.abs(epi) / np.abs(esi)
        self.EXAngSi2 = np.unwrap(np.angle(epi2)) - np.unwrap(np.angle(esi2))
        self.EXMagSi2 = np.abs(epi2) / np.abs(esi2)

    def initial_generator1(self, mag, pha):
        for x in range(0, self.ln):
            p = mag[x] * (e ** (1j * (-pha[x])))
            eps = (n0 ** 2) * ((sin(theta)) ** 2) * (1 + (((1 - p) / (1 + p)) ** 2) * (tan(theta)) ** 2)
            a = eps.real
            b = eps.imag
            self.nA1[x] = (sqrt((a + sqrt(a ** 2 + (b ** 2))) / 2)).real
            self.nA1[x] -= 1j * (b / (2 * self.nA1[x]))

    def initial_generator2(self, mag, pha):
        for x in range(0, self.ln):
            theta1 = theta_trans(self.nA1[x])
            p = mag[x] * (e ** (1j * (-pha[x])))
            eps = (self.nA1[x] ** 2) * ((sin(theta1)) ** 2) * (1 + (((1 - p) / (1 + p)) ** 2) * (tan(theta1)) ** 2)
            a = eps.real
            b = eps.imag
            self.nA2[x] = (sqrt((a + sqrt(a ** 2 + (b ** 2))) / 2)).real
            self.nA2[x] -= 1j * (b / (2 * self.nA2[x]))

    def attenuation1(self, n, f):  # /Returns attenuation factor for a given complex refractive index and frequency\
        s0 = sqrt(1 - ((n0 * sin(theta)) / n) ** 2)
        self.l1 = self.initial_l1 / s0
        return e ** (-1j * 2 * pi * f * n * self.l1 / c)  # TODO: Double Check if 4 should be 2

    def h_theory1p(self, f, n1, n2n):  # /Returns the theoretical Transfer function ratio
        # for a given frequency and complex
        # refractive index\
        p1 = self.attenuation1(n1, f) ** 2

        rp0 = rp12(n0, n1)

        rp1 = rp12(n1, n0)

        rp2 = rp12(n1, n2n)

        tp0 = (n0 / n1) * (rp0 + 1)

        ap1 = p1 * tp0 * tp0 * rp2

        ap2 = 1 - p1 * rp1 * rp2

        h2 = rp0 + ap1 / ap2

        return h2

    def h_theory1s(self, f, n1, n2n):  # /Returns the theoretical Transfer function ratio
        # for a given frequency and complex
        # refractive index\
        p1 = self.attenuation1(n1, f) ** 2

        rs0 = rs12(n0, n1)

        rs1 = rs12(n1, n0)

        rs2 = rs12(n1, n2n)

        ts0 = rs0 + 1

        as1 = p1 * ts0 * ts0 * rs2

        as2 = 1 - p1 * rs1 * rs2

        h1 = rs0 + as1 / as2

        return h1

    def err1(self, n2n):  # Returns the error between the theoretical transfer function ratio and extracted transfer
        # function ratio after changing the complex refractive index at the currently indicated
        # index to a given complex refractive index
        self.nA2t = np.copy(self.nA2)
        self.nA2t[self.current1] = n2n[0] + 1j * n2n[1]

        htp = np.zeros(self.ln, dtype=complex)
        hts = np.zeros(self.ln, dtype=complex)
        for q in range(0, self.current1 + 1):
            htap = self.h_theory1p(self.FreqS1[q], self.nA1[q], self.nA2t[q])
            htas = self.h_theory1s(self.FreqS1[q], self.nA1[q], self.nA2t[q])
            hta_r = float(htap.real)  # /
            hta_i = float(htap.imag)  # numpy problem workaround
            htp[q] = hta_r + 1j * hta_i  # \
            hta_r = float(htas.real)  # /
            hta_i = float(htas.imag)  # numpy problem workaround
            hts[q] = hta_r + 1j * hta_i  # \

        htm = np.abs(htp[self.current1]) / np.abs(hts[self.current1])
        htp2 = np.unwrap(np.angle(htp))[self.current1] - np.unwrap(np.angle(hts))[self.current1]

        mm = self.EXMagS1[self.current1] - htm
        aa = self.EXAngS1[self.current1] - htp2

        er = np.abs(mm) + np.abs(aa)
        return er

    def err2(self, n2n):  # Returns the error between the theoretical transfer function ratio and extracted transfer
        # function ratio after changing the complex refractive index at the currently indicated
        # index to a given complex refractive index
        self.nA2t = np.copy(self.nA2)
        self.nA2t[self.current1] = n2n[0] + 1j * n2n[1]

        htp = np.zeros(self.ln, dtype=complex)
        hts = np.zeros(self.ln, dtype=complex)
        for q in range(0, self.current1 + 1):
            htap = self.h_theory1p(self.FreqS1[q], self.nA1[q], self.nA2t[q])
            htas = self.h_theory1s(self.FreqS1[q], self.nA1[q], self.nA2t[q])
            hta_r = float(htap.real)  # /
            hta_i = float(htap.imag)  # numpy problem workaround
            htp[q] = hta_r + 1j * hta_i  # \
            hta_r = float(htas.real)  # /
            hta_i = float(htas.imag)  # numpy problem workaround
            hts[q] = hta_r + 1j * hta_i  # \

        htm = np.abs(hts[self.current1]) / np.abs(htp[self.current1])
        htp2 = np.unwrap(np.angle(hts))[self.current1] - np.unwrap(np.angle(htp))[self.current1]

        mm = self.EXMagS2[self.current1] - htm
        aa = self.EXAngS2[self.current1] - htp2

        er = np.abs(mm) + np.abs(aa)
        return er

    def err3(self, n2n):  # Returns the error between the theoretical transfer function ratio and extracted transfer
        # function ratio after changing the complex refractive index at the currently indicated
        # index to a given complex refractive index
        self.nA2t = np.copy(self.nA2)
        self.nA2t[self.current1] = n2n[0] + 1j * n2n[1]

        htp = np.zeros(self.ln, dtype=complex)
        hts = np.zeros(self.ln, dtype=complex)
        for q in range(0, self.current1 + 1):
            htap = self.h_theory1p(self.FreqS1[q], self.nA1[q], self.nA2t[q])
            htas = self.h_theory1s(self.FreqS1[q], self.nA1[q], self.nA2t[q])
            hta_r = float(htap.real)  # /
            hta_i = float(htap.imag)  # numpy problem workaround
            htp[q] = hta_r + 1j * hta_i  # \
            hta_r = float(htas.real)  # /
            hta_i = float(htas.imag)  # numpy problem workaround
            hts[q] = hta_r + 1j * hta_i  # \

        htm = np.abs(hts[self.current1])
        htp2 = np.unwrap(np.angle(hts))[self.current1]

        mm = np.abs(self.es)[self.current1] - htm
        aa = np.unwrap(np.angle(self.es))[self.current1] - htp2

        er = np.abs(mm) + np.abs(aa)
        return er

    def smoothness1(self, thick):  # Returns the smoothness of the complex refractive functions after changing the
        # sample thickness to a given quantity
        nmin = Array('d', range(len(self.FreqS1)))
        kmin = Array('d', range(len(self.FreqS1)))
        lt = Value('d', thick)
        recur2k(self.FreqS1, self.es, self.ep, self.esi, self.epi, self.esi2, self.epi2, nmin, kmin, lt)

        # bb = 2
        #
        # nkep = np.zeros(len(self.FreqS1))
        # kkep = np.zeros(len(self.FreqS1))
        # ksi = int(np.log10(np.abs(kmin[1])))
        # for lm in range(0, len(self.FreqS1)):
        #     nkep[lm] = round(nmin[lm], 3)
        #     kkep[lm] = round(-kmin[lm]/(10**ksi), 3)*(10**ksi)
        #
        # nkep = lulu(bb, nkep)
        # kkep = lulu(bb, kkep)

        # for lm in range(bb, len(self.FreqS1) - bb):
        #     nmin[lm] = nkep[lm]
        #     kmin[lm] = -kkep[lm]

        #
        # recur3(self.FreqS1, self.es, self.ep, nmin, kmin, lt)
        # #
        # bb = 2
        #
        # nkep = np.zeros(len(self.FreqS1))
        # kkep = np.zeros(len(self.FreqS1))
        # for lm in range(0, len(self.FreqS1)):
        #     nkep[lm] = round(nmin[lm], 3)
        #     kkep[lm] = round(-kmin[lm]/(10**ksi), 3)*(10**ksi)
        #
        # nkep = lulu(bb, nkep)
        # kkep = lulu(bb, kkep)
        #
        # for lm in range(bb, len(self.FreqS1) - bb):
        #     nmin[lm] = nkep[lm]
        #     kmin[lm] = -kkep[lm]

        # recur3(self.FreqS1, self.es, self.ep, nmin, kmin, lt)

        self.nA2 = np.zeros(len(self.FreqS1), dtype=complex)
        for zzk in range(0, len(self.FreqS1)):
            self.nA2[zzk] = nmin[zzk] + 1j * abs(kmin[zzk])
        tt = 0
        for m in range(3, self.ln - 3):  # Skip first element due to unstable nature
            de = np.abs(float(self.nA1[m - 1].real) - float(self.nA1[m].real)) \
                 + np.abs(float(self.nA1[m - 1].imag) - float(self.nA1[m].imag))
            tt += de
        return tt

    def call1(self):
        # /\ Use minimisation function to find the thickness of the sample
        shift = np.log10(self.initial_l1)
        if shift < int(shift):
            shift = 10 ** (int(shift) - 2)
        else:
            shift = 10 ** int(shift - 1)
        res = minimize_scalar(self.smoothness1, bounds=(self.initial_l1 - shift, self.initial_l1 + shift),
                              method='bounded', options={'xatol': 1e-8})

        self.initial_l1 = float(res.x)
        # \/
        return self.initial_l1

    def call2(self):
        # /\ Use minimisation function to extract the complex refractive index of the sample
        self.initial_generator1(self.EXMagSi, self.EXAngSi)
        # self.initial_generator1(self.EXMagS1, self.EXAngS1)
        # ka = 0
        # na = 0
        # for i in range(0, self.ln):
        #     na += self.nA1[i].real / self.ln
        #     ka += self.nA1[i].imag / self.ln
        # self.nA1 = (na + 1j * ka) * np.ones(self.ln, dtype=complex)

        self.current1 = 0

        for zk in range(0, self.ln):
            self.initSet1[0] = self.nA2[zk].real
            self.initSet1[1] = self.nA2[zk].imag
            res = minimize(self.err1, self.initSet1, method='nelder-mead',
                           options={'xtol': 1e-8, 'maxiter': 1000, 'disp': False})
            self.nA2[zk] = res.x[0] + 1j * res.x[1]
            self.current1 += 1

        return self.nA2

    def get_ln(self):
        return self.ln

    def set_len(self, leng):
        self.initial_l1 = leng
        self.l1 = self.initial_l1 / cos(theta)


# class Worker3k:


def recur(f, S, P, Si, Pi, l):
    Work = Worker2()
    Work.set_input(f, S, P, Si, Pi)
    Work.set_len(l.value)
    l.value = Work.call1()


def recur2(f, S, P, Si, Pi, n, k, l):
    if len(f) > 80:
        f_low = np.zeros(int(len(f)/2))
        f_high = np.zeros(int(len(f)/2))
        S_low = np.zeros(int(len(f) / 2), dtype=complex)
        S_high = np.zeros(int(len(f) / 2), dtype=complex)
        P_low = np.zeros(int(len(f) / 2), dtype=complex)
        P_high = np.zeros(int(len(f) / 2), dtype=complex)
        n_low = Array('d', range(int(len(f) / 2)))
        n_high = Array('d', range(int(len(f) / 2)))
        k_low = Array('d', range(int(len(f) / 2)))
        k_high = Array('d', range(int(len(f) / 2)))

        S_lowi = np.zeros(int(len(f) / 2), dtype=complex)
        S_highi = np.zeros(int(len(f) / 2), dtype=complex)
        P_lowi = np.zeros(int(len(f) / 2), dtype=complex)
        P_highi = np.zeros(int(len(f) / 2), dtype=complex)
        for zz in range(0, int(len(f) / 2)):
            f_low[zz] = f[zz]
            S_low[zz] = S[zz]
            P_low[zz] = P[zz]
            f_high[zz] = f[zz + int(len(f) / 2)]
            S_high[zz] = S[zz + int(len(f) / 2)]
            P_high[zz] = P[zz + int(len(f) / 2)]

            S_lowi[zz] = Si[zz]
            P_lowi[zz] = Pi[zz]
            S_highi[zz] = Si[zz + int(len(f) / 2)]
            P_highi[zz] = Pi[zz + int(len(f) / 2)]
        low = Process(target=recur2, args=(f_low, S_low, P_low, S_lowi, P_lowi, n_low, k_low, l))
        high = Process(target=recur2, args=(f_high, S_high, P_high, S_highi, P_highi, n_high, k_high, l))
        low.start()
        high.start()
        low.join()
        high.join()

        for zz in range(0, int(len(f) / 2)):
            n[zz] = n_low[zz]
            k[zz] = k_low[zz]
            n[zz + int(len(f) / 2)] = n_high[zz]
            k[zz + int(len(f) / 2)] = k_high[zz]
    else:
        Work = Worker2()
        Work.set_input(f, S, P, Si, Pi)
        Work.set_len(l.value)
        ntemp = Work.call2()
        for zz in range(0, len(f)):
            n[zz] = ntemp[zz].real
            k[zz] = ntemp[zz].imag


# def recur3(f, S, P, n, k, l):
#     if len(f) > 80:
#         f_low = np.zeros(int(len(f)/2))
#         f_high = np.zeros(int(len(f)/2))
#         S_low = np.zeros(int(len(f) / 2), dtype=complex)
#         S_high = np.zeros(int(len(f) / 2), dtype=complex)
#         P_low = np.zeros(int(len(f) / 2), dtype=complex)
#         P_high = np.zeros(int(len(f) / 2), dtype=complex)
#         n_low = Array('d', range(int(len(f) / 2)))
#         n_high = Array('d', range(int(len(f) / 2)))
#         k_low = Array('d', range(int(len(f) / 2)))
#         k_high = Array('d', range(int(len(f) / 2)))
#         for zz in range(0, int(len(f) / 2)):
#             f_low[zz] = f[zz]
#             S_low[zz] = S[zz]
#             P_low[zz] = P[zz]
#             n_low[zz] = n[zz]
#             k_low[zz] = k[zz]
#             f_high[zz] = f[zz + int(len(f) / 2)]
#             S_high[zz] = S[zz + int(len(f) / 2)]
#             P_high[zz] = P[zz + int(len(f) / 2)]
#             n_high[zz] = n[zz + int(len(f) / 2)]
#             k_high[zz] = k[zz + int(len(f) / 2)]
#         low = Process(target=recur3, args=(f_low, S_low, P_low, n_low, k_low, l))
#         high = Process(target=recur3, args=(f_high, S_high, P_high, n_high, k_high, l))
#         low.start()
#         high.start()
#         low.join()
#         high.join()
#
#         for zz in range(0, int(len(f) / 2)):
#             n[zz] = n_low[zz]
#             k[zz] = k_low[zz]
#             n[zz + int(len(f) / 2)] = n_high[zz]
#             k[zz + int(len(f) / 2)] = k_high[zz]
#     else:
#         work2 = Worker3()
#         work2.set_input(f, S, P, n, k)
#         work2.set_len(l.value)
#         ntemp = work2.call2()
#         for zz in range(0, len(f)):
#             n[zz] = ntemp[zz].real
#             k[zz] = ntemp[zz].imag


def recurk(f, S, P, Si, Pi, Si2, Pi2, l):
    Work = Worker2k()
    Work.set_input(f, S, P, Si, Pi, Si2, Pi2)
    Work.set_len(l.value)
    l.value = Work.call1()


def recur2k(f, S, P, Si, Pi, Si2, Pi2, n, k, l):
    if len(f) > 80:
        f_low = np.zeros(int(len(f)/2))
        f_high = np.zeros(int(len(f)/2))
        S_low = np.zeros(int(len(f) / 2), dtype=complex)
        S_high = np.zeros(int(len(f) / 2), dtype=complex)
        P_low = np.zeros(int(len(f) / 2), dtype=complex)
        P_high = np.zeros(int(len(f) / 2), dtype=complex)
        n_low = Array('d', range(int(len(f) / 2)))
        n_high = Array('d', range(int(len(f) / 2)))
        k_low = Array('d', range(int(len(f) / 2)))
        k_high = Array('d', range(int(len(f) / 2)))

        S_lowi = np.zeros(int(len(f) / 2), dtype=complex)
        S_highi = np.zeros(int(len(f) / 2), dtype=complex)
        P_lowi = np.zeros(int(len(f) / 2), dtype=complex)
        P_highi = np.zeros(int(len(f) / 2), dtype=complex)
        S_lowi2 = np.zeros(int(len(f) / 2), dtype=complex)
        S_highi2 = np.zeros(int(len(f) / 2), dtype=complex)
        P_lowi2 = np.zeros(int(len(f) / 2), dtype=complex)
        P_highi2 = np.zeros(int(len(f) / 2), dtype=complex)
        for zz in range(0, int(len(f) / 2)):
            f_low[zz] = f[zz]
            S_low[zz] = S[zz]
            P_low[zz] = P[zz]
            f_high[zz] = f[zz + int(len(f) / 2)]
            S_high[zz] = S[zz + int(len(f) / 2)]
            P_high[zz] = P[zz + int(len(f) / 2)]

            S_lowi[zz] = Si[zz]
            P_lowi[zz] = Pi[zz]
            S_highi[zz] = Si[zz + int(len(f) / 2)]
            P_highi[zz] = Pi[zz + int(len(f) / 2)]
            S_lowi2[zz] = Si2[zz]
            P_lowi2[zz] = Pi2[zz]
            S_highi2[zz] = Si2[zz + int(len(f) / 2)]
            P_highi2[zz] = Pi2[zz + int(len(f) / 2)]
        low = Process(target=recur2k, args=(f_low, S_low, P_low, S_lowi, P_lowi, S_lowi2, P_lowi2, n_low, k_low, l))
        high = Process(target=recur2k, args=(f_high, S_high, P_high, S_highi, P_highi, S_highi2, P_highi2, n_high,
                                             k_high, l))
        low.start()
        high.start()
        low.join()
        high.join()

        for zz in range(0, int(len(f) / 2)):
            n[zz] = n_low[zz]
            k[zz] = k_low[zz]
            n[zz + int(len(f) / 2)] = n_high[zz]
            k[zz + int(len(f) / 2)] = k_high[zz]
    else:
        Work = Worker2k()
        Work.set_input(f, S, P, Si, Pi, Si2, Pi2)
        Work.set_len(l.value)
        ntemp = Work.call2()
        for zz in range(0, len(f)):
            n[zz] = ntemp[zz].real
            k[zz] = ntemp[zz].imag


# def recur3k(f, S, P, n, l):
#     if len(f) > 80:
#         f_low = np.zeros(int(len(f)/2))
#         f_high = np.zeros(int(len(f)/2))
#         S_low = np.zeros(int(len(f) / 2), dtype=complex)
#         S_high = np.zeros(int(len(f) / 2), dtype=complex)
#         P_low = np.zeros(int(len(f) / 2), dtype=complex)
#         P_high = np.zeros(int(len(f) / 2), dtype=complex)
#         n_low = Array('d', range(int(len(f) / 2)))
#         n_high = Array('d', range(int(len(f) / 2)))
#         for zz in range(0, int(len(f) / 2)):
#             f_low[zz] = f[zz]
#             S_low[zz] = S[zz]
#             P_low[zz] = P[zz]
#             n_low[zz] = n[zz]
#             f_high[zz] = f[zz + int(len(f) / 2)]
#             S_high[zz] = S[zz + int(len(f) / 2)]
#             P_high[zz] = P[zz + int(len(f) / 2)]
#             n_high[zz] = n[zz + int(len(f) / 2)]
#         low = Process(target=recur3k, args=(f_low, S_low, P_low, n_low, l))
#         high = Process(target=recur3k, args=(f_high, S_high, P_high, n_high, l))
#         low.start()
#         high.start()
#         low.join()
#         high.join()
#
#         for zz in range(0, int(len(f) / 2)):
#             n[zz] = n_low[zz]
#             n[zz + int(len(f) / 2)] = n_high[zz]
#
#     else:
#         work2 = Worker3k()
#         work2.set_input(f, S, P, n)
#         work2.set_len(l.value)
#         ntemp = work2.call2()
#         for zz in range(0, len(f)):
#             n[zz] = ntemp[zz].real


def start_s(mc):
    bots = Value('d', botLim)
    tops = Value('d', upLim)
    global initial_l
    gln = 60.0
    threadcount = -1

    while gln > 20:
        threadcount += 1
        threads = 2**threadcount
        starter = Worker1()
        starter.set_thread(1, threads)
        starter.set_lims(bots.value, tops.value)
        starter.set_dec(Decimal)
        starter.set_mid(middy)
        starter.set_files(file_ES, file_EP)
        gln = starter.get_ln()

    nln = gln*threads
    next2 = Array('d', range(nln))
    kext = Array('d', range(nln))

    FS = open(file_ES, 'r')
    FP = open(file_EP, 'r')

    Time = []
    ES = []
    EP = []

    for line in FS:
        if Decimal == '.':
            temp = line.split()
            Time.append(float(temp[0]) * 10**-12)
            ES.append(complex(temp[1]))
        else:
            temp = line.replace(Decimal, '.').split()
            Time.append(float(temp[0]) * 10 ** -12)
            ES.append(complex(temp[1]))
    FS.close()
    for line in FP:
        if Decimal == '.':
            temp = line.split()
            EP.append(complex(temp[1]))
        else:
            temp = line.replace(Decimal, '.').split()
            EP.append(complex(temp[1]))
    FP.close()

    NFFT = 2 ** (next_pow2(len(Time)))
    Freq = (1 / abs(Time[1] - Time[0])) * (np.arange(NFFT) - NFFT / 2) / NFFT

    # start2 = 0
    # for i in range(0, len(Time)):
    #     if Time[i] >= 7.17 * 10 ** -12:
    #         start2 = i
    #         break
    #
    # stop2 = 0
    # for i in range(start2, len(Time)):
    #     if Time[i] > 14.59 * 10 ** -12:
    #         stop2 = i
    #         break
    start2 = 0
    for i in range(0, len(Time)):
        if Time[i] >= -1.6 * 10 ** -12:
            start2 = i
            break

    stop2 = 0
    for i in range(start2, len(Time)):
        if Time[i] > 2.0 * 10 ** -12:
            stop2 = i
            break
    bread = (stop2 - start2) / (2 * sqrt(2 * np.log(2)))
    gg = gauss(np.linspace(0, len(Time), len(Time)), b=np.argmax(np.abs(ES)), ccd=bread)

    if middy:
        yfp = np.fft.fftshift(np.fft.fft(EP, NFFT))
        yfs = np.fft.fftshift(np.fft.fft(ES, NFFT))
    else:
        yfp = np.fft.fftshift(np.fft.fft(np.fft.fftshift(EP), NFFT))
        yfs = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ES), NFFT))

    if middy:
        yfpi = np.fft.fftshift(np.fft.fft(EP*gg, NFFT))
        yfsi = np.fft.fftshift(np.fft.fft(ES*gg, NFFT))
    else:
        yfpi = np.fft.fftshift(np.fft.fft(np.fft.fftshift(EP*gg), NFFT))
        yfsi = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ES*gg), NFFT))

    # ff, aa, pp = correction()
    # start = 0
    #
    # for i in range(int(NFFT / 2), NFFT):
    #     if Freq[i] >= ff[0]:
    #         start = i
    #         break

    # y_p_freq2 = yfp[start: len(ff) + start]
    # y_p_freq2a = np.abs(y_p_freq2)
    # y_p_freq2p = np.unwrap(np.angle(y_p_freq2))
    # y_p_freq2i = yfpi[start: len(ff) + start]
    # y_p_freq2ai = np.abs(y_p_freq2i)
    # y_p_freq2pi = np.unwrap(np.angle(y_p_freq2i))
    #
    # for i in range(start, len(ff) + start):
    #     yfp[i] = aa[i - start] * y_p_freq2a[i - start] * exp(1j * (y_p_freq2p[i - start] + pp[i - start]))
    #     yfpi[i] = aa[i - start] * y_p_freq2ai[i - start] * exp(1j * (y_p_freq2pi[i - start] + pp[i - start]))

    stzo = 0
    for i in range(int(NFFT/2), NFFT):
        if Freq[i]*10**-12 >= botLim:
            stzo = i
            break
    enzo = stzo + nln
    freq = Freq[stzo:enzo]
    ess = yfs[stzo:enzo]
    epp = yfp[stzo:enzo]
    essi = yfsi[stzo:enzo]
    eppi = yfpi[stzo:enzo]
    l_ext = Value('d', initial_l)

    recur(freq, ess, epp, essi, eppi, l_ext)
    recur2(freq, ess, epp, essi, eppi, next2, kext, l_ext)
    bb = 5

    nkep = np.zeros(len(freq))
    kkep = np.zeros(len(freq))
    ksi = int(np.log10(np.abs(kext[1])))
    for lm in range(0, len(freq)):
        nkep[lm] = round(next2[lm], 3)
        kkep[lm] = round(-kext[lm]/(10**ksi), 3)*(10**ksi)

    nkep = lulu(bb, nkep)
    kkep = lulu(bb, kkep)

    for lm in range(bb, len(freq) - bb):
        next2[lm] = nkep[lm]
        kext[lm] = -kkep[lm]

    #
    # recur2(freq, ess, epp, next2, kext, l_ext)
    # # #
    # bb = 2
    # #
    # nkep = np.zeros(len(freq))
    # kkep = np.zeros(len(freq))
    # for lm in range(0, len(freq)):
    #     nkep[lm] = round(next2[lm], 3)
    #     kkep[lm] = round(-kext[lm]/(10**ksi), 3)*(10**ksi)
    # #
    # nkep = lulu(bb, nkep)
    # kkep = lulu(bb, kkep)
    # #
    # for lm in range(bb, len(freq) - bb):
    #     next2[lm] = nkep[lm]
    #     kext[lm] = -kkep[lm]

    # recur3(freq, ess, epp, next2, kext, l_ext)
    #
    # bb = 3
    #
    # nkep = np.zeros(len(freq))
    # kkep = np.zeros(len(freq))
    # for lm in range(0, len(freq)):
    #     nkep[lm] = next2[lm]
    #     kkep[lm] = -kext[lm]
    #
    # nkep = lulu(bb, nkep)
    # kkep = lulu(bb, kkep)
    #
    # for lm in range(bb, len(freq) - bb):
    #     next2[lm] = nkep[lm]
    #     kext[lm] = -kkep[lm]
    # recur3(freq, ess, epp, next2, kext, l_ext)
    global freqs, nres, kres
    #
    nres = np.zeros(nln)
    kres = np.zeros(nln)
    freqs = np.zeros(nln)
    #
    for k in range(0, nln):
        freqs[k] = freq[k]
        nres[k] = next2[k]
        kres[k] = -kext[k]
        
    initial_l = l_ext.value
    mc.thick.setText(str(initial_l * 10 ** 3))
    nn = open('N1Ref.txt', 'r')
    fr = []
    nr = []
    ar = []
    #
    for line in nn:
        temp = line.split()
        fr.append(float(temp[0]))
        nr.append(float(temp[1]))
        ar.append(float(temp[2]))
    #
    i1 = np.min(np.where(np.asarray(fr) >= botLim))
    i2 = np.max(np.where(np.asarray(fr) <= upLim))
    fr1 = np.asarray(fr)[i1:i2]
    nr1 = np.asarray(nr)[i1:i2]
    ar1 = np.asarray(ar)[i1:i2]

    # fr1 = np.asarray(fr)*10**12
    # nr1 = np.asarray(nr)
    # ar1 = np.asarray(ar)
    #
    # ner = np.zeros(len(nres))
    # ker = np.zeros(len(kres))
    #
    # start = 0
    # for i in range(0, len(ner)):
    #     for j in range(start, len(fr1)):
    #         if fr1[j] >= freqs[i]:
    #             start = j
    #             ner[i] = 100 * np.abs(nres[i] - nr1[j]) / nr1[j]
    #             ker[i] = 100 * np.abs(kres[i] - ar1[j]) / ar1[j]
    #             break

    mc.m4.clf()
    mc.m4.plot(freqs * 10 ** -12, nres, L='Ellipsometry', xL='Frequency (THz)', yL='Refractive Index')
    mc.m4.plot(fr1, nr1, L='Input', xl=[botLim, upLim])
    # mc.m4.plot(freqs * 10 ** -12, ner, L='Refractive Index', xL='Frequency (THz)', yL='Error(%)', xl=[botLim, upLim])
    # #
    mc.m5.clf()
    mc.m5.plot(freqs * 10 ** -12, kres, L='Ellipsometry', xL='Frequency (THz)', yL='Extinction Coefficient')
    mc.m5.plot(fr1, ar1, L='Input', xl=[botLim, upLim])
    # mc.m5.plot(freqs * 10 ** -12, ker, L='Extinction coefficient', xL='Frequency (THz)', yL='Error(%)',
    #            xl=[botLim, upLim])
    mc.lbl1.hide()
    mc.btn2.show()
    global ResShow
    if not ResShow:
        mc.btn6.setText('FFT Data')
        mc.m2.hide()
        mc.toolbar2.hide()
        mc.m3.hide()
        mc.toolbar3.hide()
        mc.m4.show()
        mc.toolbar4.show()
        mc.m5.show()
        mc.toolbar5.show()
        ResShow = True


def start_k(mc):
    bots = Value('d', botLim)
    tops = Value('d', upLim)
    global initial_l
    gln = 60.0
    threadcount = -1

    while gln > 20:
        threadcount += 1
        threads = 2 ** threadcount
        starter = Worker1()
        starter.set_thread(1, threads)
        starter.set_lims(bots.value, tops.value)
        starter.set_dec(Decimal)
        starter.set_mid(middy)
        starter.set_files(file_ES, file_EP)
        gln = starter.get_ln()

    nln = gln * threads
    next2 = Array('d', range(nln))
    kext = Array('d', range(nln))

    FS = open(file_ES, 'r')
    FP = open(file_EP, 'r')

    Time = []
    ES = []
    EP = []

    for line in FS:
        if Decimal == '.':
            temp = line.split()
            Time.append(float(temp[0]) * 10 ** -12)
            ES.append(complex(temp[1]))
        else:
            temp = line.replace(Decimal, '.').split()
            Time.append(float(temp[0]) * 10 ** -12)
            ES.append(complex(temp[1]))
    FS.close()
    for line in FP:
        if Decimal == '.':
            temp = line.split()
            EP.append(complex(temp[1]))
        else:
            temp = line.replace(Decimal, '.').split()
            EP.append(complex(temp[1]))
    FP.close()

    NFFT = 2 ** (next_pow2(len(Time)))
    Freq = (1 / abs(Time[1] - Time[0])) * (np.arange(NFFT) - NFFT / 2) / NFFT

    start2 = 0
    for i in range(0, len(Time)):
        if Time[i] >= 9.1 * 10 ** -12:
            start2 = i
            break

    stop2 = 0
    for i in range(start2, len(Time)):
        if Time[i] > 12.6 * 10 ** -12:
            stop2 = i
            break

    start3 = 0
    for i in range(0, len(Time)):
        if Time[i] >= 19 * 10 ** -12:
            start3 = i
            break

    stop3 = 0
    for i in range(start2, len(Time)):
        if Time[i] > 22.6 * 10 ** -12:
            stop3 = i
            break

    bread = (stop2 - start2) / (2 * sqrt(2 * np.log(2)))
    bread2 = (stop3 - start3) / (2 * sqrt(2 * np.log(2)))
    gg = gauss(np.linspace(0, len(Time), len(Time)), b=np.argmax(np.abs(ES)), ccd=bread)
    gg2 = gauss(np.linspace(0, len(Time), len(Time)), b=np.argmax(np.abs(ES - ES*gg)), ccd=bread2)

    if middy:
        yfp = np.fft.fftshift(np.fft.fft(EP, NFFT))
        yfs = np.fft.fftshift(np.fft.fft(ES, NFFT))
    else:
        yfp = np.fft.fftshift(np.fft.fft(np.fft.fftshift(EP), NFFT))
        yfs = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ES), NFFT))

    if middy:
        yfpi = np.fft.fftshift(np.fft.fft(EP * gg, NFFT))
        yfsi = np.fft.fftshift(np.fft.fft(ES * gg, NFFT))
    else:
        yfpi = np.fft.fftshift(np.fft.fft(np.fft.fftshift(EP * gg), NFFT))
        yfsi = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ES * gg), NFFT))

    if middy:
        yfpi2 = np.fft.fftshift(np.fft.fft(EP * gg2, NFFT))
        yfsi2 = np.fft.fftshift(np.fft.fft(ES * gg2, NFFT))
    else:
        yfpi2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(EP * gg2), NFFT))
        yfsi2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ES * gg2), NFFT))

    ff, aa, pp = correction()
    start = 0

    for i in range(int(NFFT / 2), NFFT):
        if Freq[i] >= ff[0]:
            start = i
            break

    y_p_freq2 = yfp[start: len(ff) + start]
    y_p_freq2a = np.abs(y_p_freq2)
    y_p_freq2p = np.unwrap(np.angle(y_p_freq2))
    y_p_freq2i = yfpi[start: len(ff) + start]
    y_p_freq2ai = np.abs(y_p_freq2i)
    y_p_freq2pi = np.unwrap(np.angle(y_p_freq2i))

    for i in range(start, len(ff) + start):
        yfp[i] = aa[i - start] * y_p_freq2a[i - start] * exp(1j * (y_p_freq2p[i - start] + pp[i - start]))
        yfpi[i] = aa[i - start] * y_p_freq2ai[i - start] * exp(1j * (y_p_freq2pi[i - start] + pp[i - start]))

    stzo = 0
    for i in range(int(NFFT / 2), NFFT):
        if Freq[i] * 10 ** -12 >= botLim:
            stzo = i
            break
    enzo = stzo + nln
    freq = Freq[stzo:enzo]
    ess = yfs[stzo:enzo]
    epp = yfp[stzo:enzo]
    essi = yfsi[stzo:enzo]
    eppi = yfpi[stzo:enzo]
    essi2 = yfsi2[stzo:enzo]
    eppi2 = yfpi2[stzo:enzo]
    l_ext = Value('d', initial_l)

    # recur(freq, ess, epp, essi, eppi, l_ext)

    recur2k(freq, ess, epp, essi, eppi, essi2, eppi2, next2, kext, l_ext)

    bb = 2

    nkep = np.zeros(len(freq))
    kkep = np.zeros(len(freq))
    ksi = int(np.log10(np.abs(kext[1])))
    for lm in range(0, len(freq)):
        nkep[lm] = round(next2[lm], 3)
        kkep[lm] = round(-kext[lm] / (10 ** ksi), 3) * (10 ** ksi)

    nkep = lulu(bb, nkep)
    kkep = lulu(bb, kkep)

    for lm in range(bb, len(freq) - bb):
        next2[lm] = nkep[lm]
        kext[lm] = -kkep[lm]

    #
    # recur3(freq, ess, epp, next2, kext, l_ext)
    # #
    # bb = 2
    #
    # nkep = np.zeros(len(freq))
    # kkep = np.zeros(len(freq))
    # for lm in range(0, len(freq)):
    #     nkep[lm] = round(next2[lm], 3)
    #     kkep[lm] = round(-kext[lm]/(10**ksi), 3)*(10**ksi)
    #
    # nkep = lulu(bb, nkep)
    # kkep = lulu(bb, kkep)
    #
    # for lm in range(bb, len(freq) - bb):
    #     next2[lm] = nkep[lm]
    #     kext[lm] = -kkep[lm]

    # recur3(freq, ess, epp, next2, kext, l_ext)
    #
    # bb = 3
    #
    # nkep = np.zeros(len(freq))
    # kkep = np.zeros(len(freq))
    # for lm in range(0, len(freq)):
    #     nkep[lm] = next2[lm]
    #     kkep[lm] = -kext[lm]
    #
    # nkep = lulu(bb, nkep)
    # kkep = lulu(bb, kkep)
    #
    # for lm in range(bb, len(freq) - bb):
    #     next2[lm] = nkep[lm]
    #     kext[lm] = -kkep[lm]
    # recur3(freq, ess, epp, next2, kext, l_ext)
    global freqs, nres, kres
    #
    nres = np.zeros(nln)
    kres = np.zeros(nln)
    freqs = np.zeros(nln)
    #
    for k in range(0, nln):
        freqs[k] = freq[k]
        nres[k] = next2[k]
        kres[k] = -kext[k]

    initial_l = l_ext.value
    mc.thick.setText(str(initial_l * 10 ** 3))
    nn = open('GlassRef.txt', 'r')
    fr = []
    nr = []
    ar = []
    #
    for line in nn:
        temp = line.split()
        fr.append(float(temp[0]))
        nr.append(float(temp[1]))
        ar.append(float(temp[2]))
    #
    i1 = np.min(np.where(np.asarray(fr) >= botLim))
    i2 = np.max(np.where(np.asarray(fr) <= upLim))
    fr1 = np.asarray(fr)[i1:i2]
    nr1 = np.asarray(nr)[i1:i2]
    ar1 = np.asarray(ar)[i1:i2]
    mc.m4.clf()
    mc.m4.plot(freqs * 10 ** -12, nres, L='Ellipsometry', xL='Frequency (THz)', yL='Refractive Index')
    mc.m4.plot(fr1, nr1, L='Transmission', xl=[botLim, upLim])
    # #
    mc.m5.clf()
    mc.m5.plot(freqs * 10 ** -12, kres, L='Ellipsometry', xL='Frequency (THz)', yL='Extinction Coefficient')
    mc.m5.plot(fr1, ar1 * (c * 100) / (fr1 * (10 ** 12) * 4 * pi), L='Transmission', xl=[botLim, upLim])
    mc.lbl1.hide()
    mc.btn2.show()
    global ResShow
    if not ResShow:
        mc.btn6.setText('FFT Data')
        mc.m2.hide()
        mc.toolbar2.hide()
        mc.m3.hide()
        mc.toolbar3.hide()
        mc.m4.show()
        mc.toolbar4.show()
        mc.m5.show()
        mc.toolbar5.show()
        ResShow = True


def start_b(mc):
    yp = open(file_EP, 'r')
    ys = open(file_ES, 'r')

    xt = []
    ypt = []
    yst = []
    if Decimal == '.':
        for line in yp:
            temp2 = line.split('\t')
            if not temp2[0]:
                break
            xt.append(float(temp2[0]) * 10 ** -12)
            ypt.append(complex(temp2[1]))
        yp.close()

        for line in ys:
            temp2 = line.split('\t')
            if not temp2[0]:
                break
            yst.append(complex(temp2[1]))
        ys.close()

    else:
        for line in yp:
            temp2 = line.replace(Decimal, '.').split('\t')
            if not temp2[0]:
                break
            xt.append(float(temp2[0]) * 10 ** -12)
            ypt.append(complex(temp2[1]))
        yp.close()

        for line in ys:
            temp2 = line.replace(Decimal, '.').split('\t')
            if not temp2[0]:
                break
            yst.append(complex(temp2[1]))
        ys.close()

    n_fft = 2 ** (next_pow2(len(xt)))

    time = np.asarray(xt)
    ypp = np.asarray(ypt)
    yss = np.asarray(yst)

    # start2 = 0
    # for i in range(0, len(time)):
    #     if time[i] >= 9.17 * 10 ** -12:
    #         start2 = i
    #         break
    #
    # stop2 = 0
    # for i in range(start2, len(time)):
    #     if time[i] > 12.59 * 10 ** -12:
    #         stop2 = i
    #         break

    # bread = (stop2 - start2) / (2 * sqrt(2 * np.log(2)))
    # gg = gauss(np.linspace(0, len(time), len(time)), b=np.argmax(np.abs(yss)), ccd=bread)

    freq = (1 / abs(time[1] - time[0])) * (np.arange(n_fft) - n_fft / 2) / n_fft

    if middy:
        # y_p_freq = np.fft.fftshift(np.fft.fft(ypp*gg, n_fft))
        # y_s_freq = np.fft.fftshift(np.fft.fft(yss*gg, n_fft))
        y_p_freq = np.fft.fftshift(np.fft.fft(ypp, n_fft))
        y_s_freq = np.fft.fftshift(np.fft.fft(yss, n_fft))
    else:
        # y_p_freq = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ypp*gg), n_fft))
        # y_s_freq = np.fft.fftshift(np.fft.fft(np.fft.fftshift(yss*gg), n_fft))
        y_p_freq = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ypp), n_fft))
        y_s_freq = np.fft.fftshift(np.fft.fft(np.fft.fftshift(yss), n_fft))

    # ff, aa, pp = correction()
    # start = 0

    # for i in range(int(n_fft/2), n_fft):
    #     if freq[i] >= ff[0]:
    #         start = i
    #         break

    # y_p_freq2 = y_p_freq[start: len(ff)+start]
    # y_p_freq2a = np.abs(y_p_freq2)
    # y_p_freq2p = np.unwrap(np.angle(y_p_freq2))
    #
    # for i in range(start, len(ff)+start):
    #     y_p_freq[i] = aa[i-start]*y_p_freq2a[i-start]*exp(1j*(y_p_freq2p[i-start] + pp[i-start]))

    h_experiment = np.zeros(n_fft, dtype=complex)

    for i in range(0, n_fft):
        h_experiment[i] = y_p_freq[i] / y_s_freq[i]

    startzone = 0
    zone = 0
    for i in range(0, n_fft):
        if freq[i] == 0:
            startzone = i
            break
    for i in range(startzone, n_fft):
        zone += 1
        if freq[i] >= upLim * 10 ** 12:
            break
    endz = startzone + zone
    startzone2 = 0
    for i in range(startzone, endz):
        if freq[i] >= botLim * 10 ** 12:
            startzone2 = i
            break

    endz = startzone + zone
    cuts = endz - startzone2
    t_end = startzone2 + cuts
    freq_s = freq[startzone2:t_end]
    exh = h_experiment[startzone2:t_end]
    ln = len(exh)

    global freqs, nres, kres
    freqs = np.copy(freq_s)
    nres = np.zeros(ln, dtype=complex)
    kres = np.zeros(ln, dtype=complex)

    ex_ang = np.unwrap(np.angle(exh))
    ex_mag = np.abs(exh)

    for x in range(0, ln):
        p = ex_mag[x] * (e ** (-1j * (ex_ang[x])))
        eps = (n0 ** 2) * ((sin(theta)) ** 2) * (1 + (((1 - p) / (1 + p)) ** 2) * (tan(theta)) ** 2)
        a = eps.real
        b = eps.imag
        nres[x] = (sqrt((a + sqrt(a ** 2 + (b ** 2))) / 2)).real
        kres[x] = b / (2 * nres[x])

    nn = open('N1Ref.txt', 'r')
    fr = []
    nr = []
    ar = []
    # #
    for line in nn:
        temp = line.split()
        fr.append(float(temp[0]))
        nr.append(float(temp[1]))
        ar.append(float(temp[2]))
    # #
    i1 = np.min(np.where(np.asarray(fr) >= botLim))
    i2 = np.max(np.where(np.asarray(fr) <= upLim))
    fr1 = np.asarray(fr)[i1:i2]
    nr1 = np.asarray(nr)[i1:i2]
    ar1 = np.asarray(ar)[i1:i2]

    # fr1 = np.asarray(fr)*10**12
    # nr1 = np.asarray(nr)
    # ar1 = np.asarray(ar)

    # ner = np.zeros(len(nres))
    # ker = np.zeros(len(kres))

    # start = 0
    # for i in range(0, len(ner)):
    #     for j in range(start, len(fr1)):
    #         if fr1[j] >= freqs[i]:
    #             start = j
    #             ner[i] = 100 * np.abs(nres[i] - nr1[j]) / nr1[j]
    #             ker[i] = 100 * np.abs(kres[i] - ar1[j]) / ar1[j]
    #             break

    # for mnn in range(0, len(kres)):
    #     if kres[mnn] < 0:
    #         kres[mnn] = 0
    #         nres[mnn] = nres[mnn-1]

    mc.m1.clf()
    mc.m1.plot(time * 10 ** 12, yss, L='ES', xL='Time (ps)', yL='Amplitude(a.u.)')
    mc.m1.plot(time * 10 ** 12, ypp, L='EP')
    mc.m4.clf()
    mc.m4.plot(freq_s * 10 ** -12, nres, L='Ellipsometry', xL='Frequency (THz)', yL='Refractive Index')
    mc.m4.plot(fr1, nr1, L='Input', xl=[botLim, upLim])
    # mc.m4.plot(freqs * 10 ** -12, ner, L='Refractive Index', xL='Frequency (THz)', yL='Error(%)', xl=[botLim, upLim])
    #
    mc.m5.clf()
    mc.m5.plot(freq_s * 10 ** -12, kres, L='Ellipsometry', xL='Frequency (THz)', yL='Extinction Coefficient')
    # mc.m5.plot(fr1, ar1 * (c * 100) / (fr1 * (10 ** 12) * 4 * pi), L='Transmission', xl=[botLim, upLim])
    mc.m5.plot(fr1, ar1, L='Input', xl=[botLim, upLim])
    # mc.m5.plot(freqs * 10 ** -12, ker, L='Extinction coefficient', xL='Frequency (THz)', yL='Error(%)',
    #            xl=[botLim, upLim])
    mc.lbl1.hide()
    mc.btn2.show()
    global ResShow
    if not ResShow:
        mc.btn6.setText('FFT Data')
        mc.m2.hide()
        mc.toolbar2.hide()
        mc.m3.hide()
        mc.toolbar3.hide()
        mc.m4.show()
        mc.toolbar4.show()
        mc.m5.show()
        mc.toolbar5.show()
        ResShow = True


def start_d2(mc):
    yp = open(file_EP, 'r')
    ys = open(file_ES, 'r')

    xt = []
    ypt = []
    yst = []
    if Decimal == '.':
        for line in yp:
            temp2 = line.split('\t')
            if not temp2[0]:
                break
            xt.append(float(temp2[0]) * 10 ** -12)
            ypt.append(complex(temp2[1]))
        yp.close()

        for line in ys:
            temp2 = line.split('\t')
            if not temp2[0]:
                break
            yst.append(complex(temp2[1]))
        ys.close()

    else:
        for line in yp:
            temp2 = line.replace(Decimal, '.').split('\t')
            if not temp2[0]:
                break
            xt.append(float(temp2[0]) * 10 ** -12)
            ypt.append(complex(temp2[1]))
        yp.close()

        for line in ys:
            temp2 = line.replace(Decimal, '.').split('\t')
            if not temp2[0]:
                break
            yst.append(complex(temp2[1]))
        ys.close()

    n_fft = 2 ** (next_pow2(len(xt)))

    time = np.asarray(xt)
    ypp = np.asarray(ypt)
    yss = np.asarray(yst)

    start2 = 0
    for i in range(0, len(time)):
        if time[i] >= -1.6 * 10 ** -12:
            start2 = i
            break

    stop2 = 0
    for i in range(start2, len(time)):
        if time[i] > 2.0 * 10 ** -12:
            stop2 = i
            break

    start3 = 0
    for i in range(0, len(time)):
        if time[i] >= 9.8 * 10 ** -12:
            start3 = i
            break

    stop3 = 0
    for i in range(start2, len(time)):
        if time[i] > 13.4 * 10 ** -12:
            stop3 = i
            break

    bread = (stop2 - start2) / (2 * sqrt(2 * np.log(2)))
    bread2 = (stop3 - start3) / (2 * sqrt(2 * np.log(2)))
    gg = gauss(np.linspace(0, len(time), len(time)), b=np.argmax(np.abs(yss)), ccd=bread)
    gg2 = gauss(np.linspace(0, len(time), len(time)), b=np.argmax(np.abs(yss - yss*gg)), ccd=bread2)

    freq = (1 / abs(time[1] - time[0])) * (np.arange(n_fft) - n_fft / 2) / n_fft

    if middy:
        y_p_freq1 = np.fft.fftshift(np.fft.fft(ypp*gg, n_fft))
        y_s_freq1 = np.fft.fftshift(np.fft.fft(yss*gg, n_fft))
    else:
        y_p_freq1 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ypp*gg), n_fft))
        y_s_freq1 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(yss*gg), n_fft))

    if middy:
        y_p_freq2 = np.fft.fftshift(np.fft.fft(ypp * gg2, n_fft))
        y_s_freq2 = np.fft.fftshift(np.fft.fft(yss*gg2, n_fft))
    else:
        y_p_freq2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ypp * gg2), n_fft))
        y_s_freq2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(yss*gg2), n_fft))

    # ff, aa, pp = correction()
    # start = 0
    #
    # for i in range(int(n_fft / 2), n_fft):
    #     if freq[i] >= ff[0]:
    #         start = i
    #         break

    # cy_p_freq21 = y_p_freq1[start: len(ff) + start]
    # cy_p_freq21a = np.abs(cy_p_freq21)
    # cy_p_freq21p = np.unwrap(np.angle(cy_p_freq21))
    #
    # cy_p_freq22 = y_p_freq2[start: len(ff) + start]
    # cy_p_freq22a = np.abs(cy_p_freq22)
    # cy_p_freq22p = np.unwrap(np.angle(cy_p_freq22))
    #
    # for i in range(start, len(ff) + start):
    #     y_p_freq1[i] = aa[i - start] * cy_p_freq21a[i - start] * exp(1j * (cy_p_freq21p[i - start] + pp[i - start]))
    #     y_p_freq2[i] = aa[i - start] * cy_p_freq22a[i - start] * exp(1j * (cy_p_freq22p[i - start] + pp[i - start]))

    startzone = 0
    zone = 0
    for i in range(0, n_fft):
        if freq[i] == 0:
            startzone = i
            break
    for i in range(startzone, n_fft):
        zone += 1
        if freq[i] >= upLim * 10 ** 12:
            break
    endz = startzone + zone
    startzone2 = 0
    for i in range(startzone, endz):
        if freq[i] >= botLim * 10 ** 12:
            startzone2 = i
            break

    endz = startzone + zone
    cuts = endz - startzone2
    t_end = startzone2 + cuts
    freq_s = freq[startzone2:t_end]
    y_p_freq1n = y_p_freq1[startzone2:t_end]
    y_s_freq1n = y_s_freq1[startzone2:t_end]
    y_p_freq2n = y_p_freq2[startzone2:t_end]
    y_s_freq2n = y_s_freq2[startzone2:t_end]
    ln = len(freq_s)

    global freqs, nres, kres
    freqs = np.copy(freq_s)
    nres1 = np.zeros(ln, dtype=complex)
    kres1 = np.zeros(ln, dtype=complex)
    nres2 = np.zeros(ln, dtype=complex)
    kres2 = np.zeros(ln, dtype=complex)

    theta2 = np.zeros(ln, dtype=complex)

    ex_ang1 = np.unwrap(np.angle(y_p_freq1n)) - np.unwrap(np.angle(y_s_freq1n))
    ex_mag1 = np.abs(y_p_freq1n)/np.abs(y_s_freq1n)

    for x in range(0, ln):
        p = ex_mag1[x] * (e ** (-1j * (-ex_ang1[x])))
        eps = (n0 ** 2) * ((sin(theta)) ** 2) * (1 + (((1 - p) / (1 + p)) ** 2) * (tan(theta)) ** 2)
        a = eps.real
        b = eps.imag
        nres1[x] = (sqrt((a + sqrt(a ** 2 + (b ** 2))) / 2)).real
        kres1[x] = -b / (2 * nres1[x])

    ts1 = np.zeros(ln, dtype=complex)
    tp1 = np.zeros(ln, dtype=complex)
    for x in range(0, ln):
        theta2[x] = theta_trans(nres1[x] + 1j*kres1[x])
        ts1[x] = ts12(n0, nres1[x] - 1j*kres1[x])
        tp1[x] = tp12(n0, nres1[x] - 1j*kres1[x])
        y_p_freq2n[x] = y_p_freq2n[x] / (tp1[x]*tp1[x])
        y_s_freq2n[x] = y_s_freq2n[x] / (ts1[x] * ts1[x])

    ex_ang2 = np.unwrap(np.angle(y_p_freq2n)) - np.unwrap(np.angle(y_s_freq2n))
    ex_mag2 = np.abs(y_p_freq2n)/np.abs(y_s_freq2n)

    for x in range(0, ln):
        p = ex_mag2[x] * (e ** (1j * (-ex_ang2[x])))
        eps = ((nres1[x] + 1j*kres1[x]) ** 2) * ((sin(theta2[x])) ** 2) * (1 + (((1 - p) / (1 + p)) ** 2) * (tan(theta2[x])) ** 2)
        a = eps.real
        b = eps.imag
        nres2[x] = (sqrt((a + sqrt(a ** 2 + (b ** 2))) / 2)).real
        kres2[x] = b / (2 * nres2[x])

    nn = open('N2Ref.txt', 'r')
    fr2 = []
    nr2 = []
    ar2 = []
    # #
    for line in nn:
        temp = line.split()
        fr2.append(float(temp[0]))
        nr2.append(float(temp[1]))
        ar2.append(float(temp[2]))
    # #
    i1 = np.min(np.where(np.asarray(fr2) >= botLim))
    i2 = np.max(np.where(np.asarray(fr2) <= upLim))
    fr22 = np.asarray(fr2)[i1:i2] * 10 ** 12
    nr22 = np.asarray(nr2)[i1:i2]
    # ar1 = np.asarray(ar)
    kr22 = np.asarray(ar2)[i1:i2]

    # fr22 = np.asarray(fr2) * 10 ** 12
    # nr22 = np.asarray(nr2)
    # # ar1 = np.asarray(ar)
    # kr22 = np.asarray(ar2)
    #
    # ner = np.zeros(len(nres2))
    # ker = np.zeros(len(kres2))
    #
    # start = 0
    # for i in range(0, len(ner)):
    #     for j in range(start, len(fr2)):
    #         if fr22[j] >= freq_s[i]:
    #             start = j
    #             ner[i] = 100 * np.abs(nres2[i] - nr22[j]) / nr22[j]
    #             ker[i] = 100 * np.abs(kres2[i] - kr22[j]) / kr22[j]
    #             break

    mc.m1.clf()
    # mc.m1.plot(time * 10 ** 12, yss*gg, L='ES', xL='Time (ps)', yL='Amplitude(a.u.)')
    # mc.m1.plot(time * 10 ** 12, ypp*gg, L='EP')
    mc.m1.plot(time * 10 ** 12, yss, L='ES', xL='Time (ps)', yL='Amplitude(a.u.)')
    mc.m1.plot(time * 10 ** 12, ypp, L='EP')
    # mc.m1.plot(time * 10 ** 12, yss*gg2, L='ES2')
    # mc.m1.plot(time * 10 ** 12, ypp*gg2, L='EP2')
    # mc.m1.plot(time * 10 ** 12, yss, L='ES0')
    # mc.m1.plot(time * 10 ** 12, ypp, L='EP0')
    # mc.m1.semilogy(freq_s * 10 ** -12, np.abs(y_p_freq2n), L='Ellipsometry',
    #  xL='Frequency (THz)', yL='Refractive Index')
    # mc.m1.semilogy(freq_s * 10 ** -12, np.abs(y_s_freq2n), L='Transmission', xl=[botLim, upLim])
    # mc.m1.plot(freq_s * 10 ** -12, np.unwrap(np.angle(y_p_freq2n)), L='Ellipsometry',
    # xL='Frequency (THz)', yL='Refractive Index')
    # mc.m1.plot(ff * 10 ** -12, aa, L='Ellipsometry', xL='Frequency (THz)',
    #            yL='Refractive Index')
    # mc.m1.plot(freq_s * 10 ** -12, np.unwrap(np.angle(y_s_freq2n)), L='Transmission', xl=[botLim, upLim])
    mc.m4.clf()
    # mc.m4.plot(freq_s * 10 ** -12, nres1, L='Layer1', xL='Frequency (THz)', yL='Refractive Index')
    # mc.m4.plot(freq_s * 10 ** -12, nres2, L='Layer2', xl=[botLim, upLim])
    mc.m4.plot(freq_s * 10 ** -12, nres2, L='Layer2', xL='Frequency (THz)', yL='Refractive Index')
    mc.m4.plot(fr22 * 10 ** -12, nr22, L='Layer2O', xl=[botLim, upLim])
    # mc.m4.plot(freq_s * 10 ** -12, ner, L='Refractive Index', xL='Frequency (THz)', yL='Error(%)', xl=[botLim, upLim])
    #
    mc.m5.clf()
    # mc.m5.plot(freq_s * 10 ** -12, kres1, L='Layer1', xL='Frequency (THz)', yL='Extinction Coefficient')
    # mc.m5.plot(freq_s * 10 ** -12, kres2, L='Layer2', xl=[botLim, upLim])
    mc.m5.plot(freq_s * 10 ** -12, kres2, L='Layer2', xL='Frequency (THz)', yL='Extinction Coefficient')
    mc.m5.plot(fr22 * 10 ** -12, kr22, L='Layer2O', xl=[botLim, upLim])
    # mc.m5.plot(freq_s * 10 ** -12, ker, L='Extinction coefficient', xL='Frequency (THz)', yL='Error(%)',
    #            xl=[botLim, upLim])
    mc.lbl1.hide()
    mc.btn2.show()
    global ResShow
    if not ResShow:
        mc.btn6.setText('FFT Data')
        mc.m2.hide()
        mc.toolbar2.hide()
        mc.m3.hide()
        mc.toolbar3.hide()
        mc.m4.show()
        mc.toolbar4.show()
        mc.m5.show()
        mc.toolbar5.show()
        ResShow = True


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_widget = QWidget(self)
        self.ret = 's'
        hbox1 = QHBoxLayout()
        hbox1.addStretch(1)
        hbox2 = QHBoxLayout()
        hbox2.addStretch(1)
        vbox_w1 = QVBoxLayout()
        vbox_w1.addStretch(1)
        vbox_w2 = QVBoxLayout()
        vbox_w2.addStretch(1)
        vbox_w3 = QVBoxLayout()
        vbox_w3.addStretch(1)
        vbox1 = QVBoxLayout()
        vbox1.addStretch(1)
        vbox2 = QVBoxLayout()
        vbox2.addStretch(1)
        vbox3 = QVBoxLayout()
        vbox3.addStretch(1)
        vbox0 = QVBoxLayout()
        vbox0.addStretch(1)
        self.m1 = PlotCanvas(self.main_widget, width=5, height=4)
        toolbar1 = NavigationToolbar(self.m1, self.main_widget)
        self.m2 = PlotCanvas(self.main_widget, width=5, height=4)
        self.toolbar2 = NavigationToolbar(self.m2, self.main_widget)
        self.m3 = PlotCanvas(self.main_widget, width=5, height=4)
        self.toolbar3 = NavigationToolbar(self.m3, self.main_widget)
        self.m4 = PlotCanvas(self.main_widget, width=5, height=4)
        self.toolbar4 = NavigationToolbar(self.m4, self.main_widget)
        self.m5 = PlotCanvas(self.main_widget, width=5, height=4)
        self.toolbar5 = NavigationToolbar(self.m5, self.main_widget)
        self.m4.hide()
        self.toolbar4.hide()
        self.m5.hide()
        self.toolbar5.hide()
        btn1 = QPushButton('Check', self)
        btn1.clicked.connect(self.check)
        btn1.resize(btn1.sizeHint())
        vbox_w1.addWidget(btn1)
        self.btn2 = QPushButton('Run', self)
        self.btn2.clicked.connect(self.starter)
        self.btn2.resize(btn1.sizeHint())
        self.lbl1 = QLabel()
        self.lbl1.setText('Running')
        self.lbl1.hide()
        self.lbl2 = QLabel()
        self.lbl2.setText('Thickness (mm)')
        self.lbl3 = QLabel()
        self.lbl3.setText('Lower Limit (THz)')
        self.lbl4 = QLabel()
        self.lbl4.setText('Upper Limit (THz)')
        self.lbl5 = QLabel()
        self.lbl5.setText('Decimal Indicator')
        vbox_w1.addWidget(self.btn2)
        vbox_w1.addWidget(self.lbl1)
        self.combo = QComboBox(self)
        self.combo.addItem('Single Layer Isotropic')
        self.combo.addItem('Single Layer Isotropic Low k')
        self.combo.addItem('Bulk Isotropic')
        self.combo.addItem('Double Layer Isotropic')
        self.combo.activated[str].connect(self.alg_choice)
        self.btn3 = QPushButton('No Shift', self)
        self.btn3.setToolTip('Set whether or not a FFTshift must be performed on the inputted data'
                             'prior to the FFT')
        self.btn3.clicked.connect(self.mid_tog)
        self.btn3.resize(btn1.sizeHint())
        vbox_w1.addWidget(self.btn3)
        self.btn4 = QPushButton('Open EP', self)
        self.btn4.clicked.connect(self.open_p)
        self.btn4.resize(btn1.sizeHint())
        self.btn5 = QPushButton('Open ES', self)
        self.btn5.clicked.connect(self.open_s)
        self.btn5.resize(btn1.sizeHint())
        self.btn6 = QPushButton('Results', self)
        self.btn6.clicked.connect(self.reser)
        self.btn6.resize(btn1.sizeHint())
        self.btn7 = QPushButton('Open n1 Reference', self)
        self.btn7.clicked.connect(self.open_ref)
        self.btn7.resize(btn1.sizeHint())
        self.btn7.hide()
        self.btn8 = QPushButton('Save', self)
        self.btn8.clicked.connect(self.save)
        self.btn8.resize(btn1.sizeHint())
        vbox_w1.addWidget(self.btn6)
        self.ref = QLineEdit(self)
        self.ref.setToolTip('File containing first layer refractive index')
        self.ref.setText(RefDat)
        self.ref.hide()
        self.savl = QLineEdit(self)
        self.savl.setToolTip('File to which results will be saved')
        self.savl.setText(saveSpace)
        self.f_ep = QLineEdit(self)
        self.f_ep.setToolTip('File containing measured data for p-polarized light')
        self.f_ep.setText(file_EP)
        self.f_es = QLineEdit(self)
        self.f_es.setToolTip('File containing measured data for s-polarized light')
        self.f_es.setText(file_ES)
        self.deci = QLineEdit(self)
        self.deci.setText(Decimal)
        self.thick = QLineEdit(self)
        self.thick.setText('0.0')
        self.bot = QLineEdit(self)
        self.bot.setText('0.5')
        self.top = QLineEdit(self)
        self.top.setText('2.0')
        self.ref.textChanged[str].connect(self.on_changed7)
        self.f_ep.textChanged[str].connect(self.on_changed1)
        self.f_es.textChanged[str].connect(self.on_changed2)
        self.deci.textChanged[str].connect(self.on_changed3)
        self.thick.textChanged[str].connect(self.on_changed4)
        self.bot.textChanged[str].connect(self.on_changed5)
        self.top.textChanged[str].connect(self.on_changed6)
        self.savl.textChanged[str].connect(self.on_changed8)
        vbox_w2.addWidget(self.savl)
        vbox_w2.addWidget(self.ref)
        vbox_w2.addWidget(self.f_ep)
        vbox_w2.addWidget(self.f_es)
        vbox_w3.addWidget(self.combo)
        vbox_w3.addWidget(self.btn8)
        vbox_w3.addWidget(self.btn7)
        vbox_w3.addWidget(self.btn4)
        vbox_w3.addWidget(self.btn5)
        vbox_w1.addWidget(self.lbl5)
        vbox_w1.addWidget(self.deci)
        vbox_w1.addWidget(self.lbl2)
        vbox_w1.addWidget(self.thick)
        vbox_w2.addWidget(self.lbl3)
        vbox_w2.addWidget(self.bot)
        vbox_w3.addWidget(self.lbl4)
        vbox_w3.addWidget(self.top)
        hbox1.addLayout(vbox_w1)
        hbox1.addLayout(vbox_w2)
        hbox1.addLayout(vbox_w3)
        vbox1.addWidget(self.m1)
        vbox1.addWidget(toolbar1)
        hbox1.addLayout(vbox1)
        vbox2.addWidget(self.m2)
        vbox2.addWidget(self.toolbar2)
        vbox2.addWidget(self.m4)
        vbox2.addWidget(self.toolbar4)
        hbox2.addLayout(vbox2)
        vbox3.addWidget(self.m3)
        vbox3.addWidget(self.toolbar3)
        vbox3.addWidget(self.m5)
        vbox3.addWidget(self.toolbar5)
        hbox2.addLayout(vbox3)
        vbox0.addLayout(hbox1)
        vbox0.addLayout(hbox2)
        self.main_widget.setLayout(vbox0)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.initui()

    def initui(self):
        self.setWindowTitle('Solve')
        self.show()

    def on_changed1(self, text):
        global file_EP
        file_EP = text

    def on_changed2(self, text):
        global file_ES
        file_ES = text

    def on_changed3(self, text):
        global Decimal
        Decimal = text

    def on_changed4(self, text):
        global initial_l
        initial_l = float(text) * 10**-3

    def on_changed5(self, text):
        global botLim
        botLim = float(text)

    def on_changed6(self, text):
        global upLim
        upLim = float(text)

    def on_changed7(self, text):
        global RefDat
        RefDat = text

    def on_changed8(self, text):
        global saveSpace
        saveSpace = text

    def mid_tog(self):
        global middy
        if middy:
            self.btn3.setText('Shift')
            middy = False
        else:
            self.btn3.setText('No Shift')
            middy = True

    def alg_choice(self, text):
        if text == 'Bulk Isotropic':
            self.lbl2.hide()
            self.thick.hide()
            if self.ret == 'd':
                self.ref.hide()
                self.btn7.hide()
            self.ret = 'b'

        elif text == 'Single Layer Isotropic':
            if self.ret == 'b':
                self.lbl2.show()
                self.thick.show()
            elif self.ret == 'd':
                self.ref.hide()
                self.btn7.hide()
            self.ret = 's'

        elif text == 'Single Layer Isotropic Low k':
            if self.ret == 'b':
                self.lbl2.show()
                self.thick.show()
            elif self.ret == 'd':
                self.ref.hide()
                self.btn7.hide()
            self.ret = 'k'

        else:
            self.ref.show()
            self.btn7.show()
            if self.ret == 'b':
                self.lbl2.show()
                self.thick.show()
            self.ret = 'd'

    def reser(self):
        global ResShow
        if ResShow:
            self.btn6.setText('Results')
            self.m4.hide()
            self.toolbar4.hide()
            self.m5.hide()
            self.toolbar5.hide()
            self.m2.show()
            self.toolbar2.show()
            self.m3.show()
            self.toolbar3.show()
            ResShow = False
        else:
            self.btn6.setText('FFT Data')
            self.m2.hide()
            self.toolbar2.hide()
            self.m3.hide()
            self.toolbar3.hide()
            self.m4.show()
            self.toolbar4.show()
            self.m5.show()
            self.toolbar5.show()
            ResShow = True

    def open_ref(self):
        global RefDat
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            RefDat = file_name
            self.ref.setText(RefDat)

    def open_p(self):
        global file_EP
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            file_EP = file_name
            self.f_ep.setText(file_EP)

    def open_s(self):
        global file_ES
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            file_ES = file_name
            self.f_es.setText(file_ES)

    def save(self):
        wr = open(saveSpace, 'w')
        for wri in range(0, len(nres)):
            wr.write(str(freqs[wri]*10**-12) + '\t' + str(nres[wri]) + '\t' + str(kres[wri]) + '\n')
        wr.close()

    def check(self):
        global initial_l, middy, upLim, botLim, Decimal, file_EP, file_ES
        yt = open(file_ES, 'r')
        tempc = yt.readline().split()[0].split(Decimal)
        yt.close()
        if len(tempc) == 1:
            self.lbl5.setStyleSheet('QLabel {color: red;}')
        else:
            self.lbl5.setStyleSheet('QLabel {color: black;}')
            if __name__ == '__main__':
                c_t = Worker1()
                c_t.set_thread(0, Threader)
                c_t.set_lims(botLim, upLim)
                c_t.set_dec(Decimal)
                c_t.set_mid(middy)
                c_t.set_files(file_ES, file_EP)

                global time_h, p_h, s_h, freq_plot, pf_plot, sf_plot, pha_p, pha_s
                time_h, p_h, s_h, freq_h, y_p_freq, y_s_freq, y_p_freqi, y_s_freqi = c_t.get_orig()
                n_fftt = len(y_p_freq)
                n_fftt2 = int(n_fftt / 2)
                freq_plot = freq_h[n_fftt2:n_fftt]
                pf_plot = y_p_freq[n_fftt2:n_fftt]
                sf_plot = y_s_freq[n_fftt2:n_fftt]

                pha_p = np.unwrap(np.angle(pf_plot))
                pha_s = np.unwrap(np.angle(sf_plot))

                ff, aa, pp = correction()
                start = 0

                for i in range(0, n_fftt2):
                    if freq_plot[i] == ff[0]:
                        start = i
                        break

                pf_plot2 = pf_plot[start:start+len(ff)]
                pf_plot3 = aa*pf_plot2
                sf_plot2 = sf_plot[start:start+len(ff)]
                pha_p2 = np.unwrap(np.angle(pf_plot2))
                pha_p3 = np.unwrap(np.angle(pf_plot2)+pp)
                pha_s2 = np.unwrap(np.angle(sf_plot2))

                self.m1.clf()
                self.m1.plot(time_h * 10 ** 12, s_h, L='ES', xL='Time(ps)', yL='Amplitude(a.u.)')
                self.m1.plot(time_h * 10 ** 12, p_h, L='EP')
                #
                # self.m2.clf()
                # self.m2.semilogy(freq_plot * 10 ** -12, np.abs(sf_plot), L='ES', xL='Frequency (THz)',
                #                  yL='Amplitude(a.u.)')
                # self.m2.semilogy(freq_plot * 10 ** -12, np.abs(pf_plot), L='EP', xl=[botLim, upLim])
                # #
                # self.m3.clf()
                # self.m3.plot(freq_plot * 10 ** -12, pha_s, L='ES', xL='Frequency (THz)', yL='Phase(Rads)')
                # self.m3.plot(freq_plot * 10 ** -12, pha_p, L='EP', xl=[botLim, upLim])
                self.m2.clf()
                self.m2.semilogy(ff * 10 ** -12, np.abs(sf_plot2), L='ES', xL='Frequency (THz)',
                                 yL='Amplitude(a.u.)')
                self.m2.semilogy(ff * 10 ** -12, np.abs(pf_plot2), L='EP', xl=[botLim, upLim])
                self.m2.semilogy(ff * 10 ** -12, np.abs(pf_plot3), L='EP - Calibrated')
                #
                self.m3.clf()
                self.m3.plot(ff * 10 ** -12, pha_s2, L='ES', xL='Frequency (THz)', yL='Phase(Rads)')
                self.m3.plot(ff * 10 ** -12, pha_p2, L='EP', xl=[botLim, upLim])
                self.m3.plot(ff * 10 ** -12, pha_p3, L='EP - Calibrated')

    def starter(self):
        global initial_l, middy, upLim, botLim, Decimal, file_EP, file_ES
        yt = open(file_ES, 'r')
        tempc = yt.readline().split()[0].split(Decimal)
        yt.close()

        if len(tempc) == 1:
            self.lbl5.setStyleSheet('QLabel {color: red;}')
        else:
            self.lbl5.setStyleSheet('QLabel {color: black;}')
            if __name__ == '__main__':
                c_t = Worker1()
                c_t.set_thread(0, Threader)
                c_t.set_lims(botLim, upLim)
                c_t.set_dec(Decimal)
                c_t.set_mid(middy)
                c_t.set_files(file_ES, file_EP)

                global time_h, p_h, s_h, freq_plot, pf_plot, sf_plot, pha_p, pha_s
                time_h, p_h, s_h, freq_h, y_p_freq, y_s_freq, y_p_freqi, y_s_freqi = c_t.get_orig()
                n_fftt = len(y_p_freq)
                n_fftt2 = int(n_fftt / 2)
                freq_plot = freq_h[n_fftt2:n_fftt]

                pf_plot = y_p_freq[n_fftt2:n_fftt]
                sf_plot = y_s_freq[n_fftt2:n_fftt]

                pha_p = np.unwrap(np.angle(pf_plot))
                pha_s = np.unwrap(np.angle(sf_plot))

                # ff, aa, pp = correction()
                # start = 0
                #
                # for i in range(0, n_fftt2):
                #     if freq_plot[i] >= ff[0]:
                #         start = i
                #         break

                # pf_plot2 = pf_plot[start:start + len(ff)]
                # pf_plot3 = aa * pf_plot2
                # sf_plot2 = sf_plot[start:start + len(ff)]
                # pha_p2 = np.unwrap(np.angle(pf_plot2))
                # pha_p3 = np.unwrap(np.angle(pf_plot2) + pp)
                # pha_s2 = np.unwrap(np.angle(sf_plot2))

                self.m1.clf()
                self.m1.plot(time_h * 10 ** 12, s_h, L='ES', xL='Time(ps)', yL='Amplitude(a.u.)')
                self.m1.plot(time_h * 10 ** 12, p_h, L='EP')
                #
                # self.m2.clf()
                # self.m2.semilogy(freq_plot * 10 ** -12, np.abs(sf_plot), L='ES', xL='Frequency (THz)',
                #                  yL='Amplitude(a.u.)')
                # self.m2.semilogy(freq_plot * 10 ** -12, np.abs(pf_plot), L='EP', xl=[botLim, upLim])
                #
                # #
                # self.m3.clf()
                # self.m3.plot(freq_plot * 10 ** -12, pha_s, L='ES', xL='Frequency (THz)', yL='Phase(Rads)')
                # self.m3.plot(freq_plot * 10 ** -12, pha_p, L='EP', xl=[botLim, upLim])

                self.m2.clf()
                self.m2.semilogy(freq_plot * 10 ** -12, np.abs(sf_plot), L='ES', xL='Frequency (THz)',
                                 yL='Amplitude(a.u.)')
                self.m2.semilogy(freq_plot * 10 ** -12, np.abs(pf_plot), L='EP', xl=[botLim, upLim])
                # self.m2.semilogy(ff * 10 ** -12, np.abs(pf_plot3), L='EP - Calibrated')
                #

                self.m3.clf()
                self.m3.plot(freq_plot * 10 ** -12, pha_s, L='ES', xL='Frequency (THz)', yL='Phase(Rads)')
                self.m3.plot(freq_plot * 10 ** -12, pha_p, L='EP', xl=[botLim, upLim])
                # self.m3.plot(ff * 10 ** -12, pha_p3, L='EP - Calibrated')
                if self.ret == 's':
                    if initial_l > 0.0:
                        self.lbl2.setStyleSheet('QLabel {color: black;}')
                        self.btn2.hide()
                        self.lbl1.show()
                        start_s(self)
                        # thread = threading.Thread(target=start_s, args=(self,))
                        # thread.daemon = True
                        # thread.start()

                    else:
                        self.lbl2.setStyleSheet('QLabel {color: red;}')

                elif self.ret == 'k':
                    if initial_l > 0.0:
                        self.lbl2.setStyleSheet('QLabel {color: black;}')
                        self.btn2.hide()
                        self.lbl1.show()
                        start_k(self)
                        # thread = threading.Thread(target=start_s, args=(self,))
                        # thread.daemon = True
                        # thread.start()

                    else:
                        self.lbl2.setStyleSheet('QLabel {color: red;}')

                elif self.ret == 'b':
                    self.lbl2.setStyleSheet('QLabel {color: black;}')
                    self.btn2.hide()
                    self.lbl1.show()
                    start_b(self)
                    # thread = threading.Thread(target=start_b, args=(self,))
                    # thread.daemon = True
                    # thread.start()

                elif self.ret == 'd':
                    if initial_l > 0.0:
                        self.lbl2.setStyleSheet('QLabel {color: black;}')
                        self.btn2.hide()
                        self.lbl1.show()
                        start_d2(self)
                        # thread = threading.Thread(target=start_d2, args=(self,))
                        # thread.daemon = True
                        # thread.start()

                    else:
                        self.lbl2.setStyleSheet('QLabel {color: red;}')


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, x, y, L=None, xL=None, yL=None, xl=None, yl=None):
        ax = self.figure.add_subplot(111)
        if not L:
            ax.plot(x, y)
        else:
            ax.plot(x, y, label=L)
            ax.legend(loc='upper right')
        if xL:
            ax.set_xlabel(xL)
        if yL:
            ax.set_ylabel(yL)
        if xl:
            ax.set_xlim(xl)
        if yl:
            ax.set_ylim(yl)
        self.draw()

    def semilogy(self, x, y, L=None, xL=None, yL=None, xl=None, yl=None):
        ax = self.figure.add_subplot(111)
        if not L:
            ax.semilogy(x, y)
        else:
            ax.semilogy(x, y, label=L)
            ax.legend(loc='upper right')
        if xL:
            ax.set_xlabel(xL)
        if yL:
            ax.set_ylabel(yL)
        if xl:
            ax.set_xlim(xl)
        if yl:
            ax.set_ylim(yl)
        self.draw()

    def clf(self):
        self.figure.clf()
        self.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
