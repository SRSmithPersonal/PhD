SolveSim_1nckLUULnk5t2QT6 - Copy6 - Copy2WithNewCorNum.py

#!/usr/bin/env python3
import sys
from numpy import *
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
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
# theta = 60.73*pi/180
# theta = 61.1516*pi/180
# theta = 60.288*pi/180
theta = 60.0*pi/180
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
    return arcsin((n0/n_out)*sin(theta))


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


def delay(n_1, d1):
    phi = (pi / 2) - theta
    alp = 105 * pi / 180 - phi / 2
    bet = 75 * pi / 180 + phi / 2

    theta1 = arcsin(n0 * sin(theta) / n_1)
    d = d1
    x1 = 77.55 * 10 ** -3
    dx = 2 * d * tan(theta1)
    AE = x1 - (x1 * tan(phi) / tan(bet))
    AC = AE - dx
    DE = x1 / cos(phi)

    B1 = sqrt(DE ** 2 + AE ** 2 - 2 * DE * AE * cos(phi))
    B2 = B1 / (dx / AC + 1)
    BC = sqrt(B2 ** 2 + AC ** 2 - B2 * AC * cos(alp))

    dB = B1 - B2
    dy2 = dB * sin(bet)
    dx2 = dB * cos(bet) + dy2 / tan(30 * pi / 180)
    y1 = (57.55 * 10 ** -3 + dx2) * tan(30 * pi / 180)
    x2 = y1 / tan(75 * pi / 180)

    B = y1 / sin(75 * pi / 180)
    A = x2 + dx2 + 57.55 * 10 ** -3
    C = (57.55 * 10 ** -3 + dx2) / cos(30 * pi / 180)

    df = (dx2 * B / (A - dx2)) / (1 + dx2 / (A - dx2))
    xe = df * cos(75 * pi / 180)
    r1 = C - dy2 / sin(30 * pi / 180)
    r2 = sqrt((A - dx2) ** 2 + (B - df) ** 2 - 2 * (A - dx2) * (B - df) * cos(75 * pi / 180))
    re = r2 - r1

    dl = DE - BC - xe - re
    return dl/c


def han(cen, le, l2):
    kk = hanning(l2)
    kai0 = zeros(le)
    for jj in range(0, len(kk)):
        kai0[cen - int(len(kk) / 2) + jj] = kk[jj]
    return kai0


def lulu(bb, arr):
    le = zeros(bb + 1)
    ue = zeros(bb + 1)

    arrt = zeros(len(arr) + 2*bb)
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
            le[lm2] = min(aa)
        arrt[lm] = max(le)

    for lm in range(bb, len(arrt)-bb):
        for lm2 in range(0, bb+1):
            aa = arrt[lm-bb + lm2: lm + lm2 + 1]
            ue[lm2] = max(aa)
        arrt[lm] = min(ue)
    nlu = copy(arrt)

    arrt = zeros(len(arr) + 2 * bb)
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
            ue[lm2] = max(aa)
        arrt[lm] = min(ue)

    for lm in range(bb, len(arrt) - bb):
        for lm2 in range(0, bb + 1):
            aa = arrt[lm - bb + lm2: lm + lm2 + 1]
            le[lm2] = min(aa)
        arrt[lm] = max(le)

    nul = copy(arrt)

    for lm in range(bb, len(arrt) - bb):
        if nlu[lm] == nul[lm]:
            arr[lm - bb] = nlu[lm]
        elif abs(arr[lm - bb] - nlu[lm]) > abs(arr[lm - bb] - nul[lm]):
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
        self.initSet1 = zeros(2)
        self.Counting = 0
        self.StartZone = 0
        self.Zone = 0
        self.StartZone2 = 0
        self.time = zeros(2)
        self.FreqS1 = zeros(2)
        self.YPP = zeros(2)
        self.YSS = zeros(2)
        self.h_experiment = zeros(2, dtype=complex)
        self.nA1 = zeros(2, dtype=complex)
        self.EXAngS1 = zeros(2)
        self.EXMagS1 = zeros(2)
        self.ln = 0
        self.freq = zeros(2)
        self.y_s_freq = zeros(2, dtype=complex)
        self.y_p_freq = zeros(2, dtype=complex)
        self.y_s_freqi = zeros(2, dtype=complex)
        self.y_p_freqi = zeros(2, dtype=complex)
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

        self.time = asarray(xt)
        self.YPP = asarray(ypt)
        self.YSS = asarray(yst)

        a1 = argmax(abs(self.YSS))
        start0 = where(self.time >= self.time[a1] - 5.0 * 10 ** -12)[0][0]
        stop0 = where(self.time >= self.time[a1] + 5.0 * 10 ** -12)[0][0]
        lo = stop0 - start0
        gg = han(a1, len(self.time), lo)

        self.freq = (1 / abs(self.time[1] - self.time[0])) * (arange(n_fft) - n_fft / 2) / n_fft

        if self.middy:
            self.y_p_freq = fft.fftshift(fft.fft(self.YPP, n_fft))
            self.y_s_freq = fft.fftshift(fft.fft(self.YSS, n_fft))
        else:
            self.y_p_freq = fft.fftshift(fft.fft(fft.fftshift(self.YPP), n_fft))
            self.y_s_freq = fft.fftshift(fft.fft(fft.fftshift(self.YSS), n_fft))

        if self.middy:
            self.y_p_freqi = fft.fftshift(fft.fft(self.YPP*gg, n_fft))
            self.y_s_freqi = fft.fftshift(fft.fft(self.YSS*gg, n_fft))
        else:
            self.y_p_freqi = fft.fftshift(fft.fft(fft.fftshift(self.YPP*gg), n_fft))
            self.y_s_freqi = fft.fftshift(fft.fft(fft.fftshift(self.YSS*gg), n_fft))

        self.h_experiment = zeros(n_fft, dtype=complex)

        for i in range(0, n_fft):
            self.h_experiment[i] = self.y_p_freq[i]/self.y_s_freq[i]

        self.StartZone2 = where(self.freq >= self.botLim * 10 ** 12)[0][0]
        endz = where(self.freq >= self.topLim * 10 ** 12)[0][0]

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
        self.nA1 = zeros(self.ln, dtype=complex)

        self.EXAngS1 = unwrap(angle(exh1))
        self.EXMagS1 = abs(exh1)

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
        self.l1 = 0
        self.current1 = 0
        self.initSet1 = zeros(2)
        self.Counting = 0
        self.FreqS1 = zeros(2)
        self.YPP = zeros(2)
        self.YSS = zeros(2)
        self.nA1 = zeros(2, dtype=complex)
        self.nA1t = zeros(2, dtype=complex)
        self.EXAngS1 = zeros(2)
        self.EXMagS1 = zeros(2)
        self.EXAngS2 = zeros(2)
        self.EXMagS2 = zeros(2)
        self.EXAngSi = zeros(2)
        self.EXMagSi = zeros(2)
        self.ln = 0
        self.freq = zeros(2)
        self.y_s_freq = zeros(2, dtype=complex)
        self.y_p_freq = zeros(2, dtype=complex)
        self.y_s_freqi = zeros(2, dtype=complex)
        self.y_p_freqi = zeros(2, dtype=complex)
        self.es = zeros(2, dtype=complex)
        self.ep = zeros(2, dtype=complex)
        self.esi = zeros(2, dtype=complex)
        self.epi = zeros(2, dtype=complex)
        self.cor_a = []
        self.cor_p = []
        self.cor_a2 = []
        self.cor_p2 = []

    def set_input(self, fr, es, ep, esi, epi, cor_a, cor_p, cor_a2, cor_p2):
        self.FreqS1 = zeros(len(fr))
        self.es = zeros(len(fr), dtype=complex)
        self.ep = zeros(len(fr), dtype=complex)
        self.esi = zeros(len(fr), dtype=complex)
        self.epi = zeros(len(fr), dtype=complex)
        exh = zeros(len(fr), dtype=complex)
        self.cor_a = cor_a
        self.cor_p = cor_p
        self.cor_a2 = cor_a2
        self.cor_p2 = cor_p2
        for wr in range(0, len(fr)):
            self.FreqS1[wr] = fr[wr]
            self.es[wr] = es[wr]
            self.ep[wr] = ep[wr]
            self.esi[wr] = esi[wr]
            self.epi[wr] = epi[wr]
            exh[wr] = ep[wr]/es[wr]
        self.ln = int(len(exh))
        self.nA1 = zeros(self.ln, dtype=complex)

        self.EXAngS1 = unwrap(angle(ep))-unwrap(angle(es)) #- cor_p
        self.EXMagS1 = (abs(ep)/abs(es))#/cor_a
        self.EXAngS2 = unwrap(angle(es)-angle(ep))  # - cor_p
        self.EXMagS2 = abs(es/ep)  # /cor_a
        self.EXAngSi = unwrap(angle(epi)) - unwrap(angle(esi))# - cor_p2
        self.EXMagSi = (abs(epi) / abs(esi))#/cor_a2

    def initial_generator1(self, mag, pha):
        for x in range(0, self.ln):
            p = mag[x] * (e ** (1j * (-pha[x])))
            eps = (n0 ** 2) * ((sin(theta)) ** 2) * (1 + (((1 - p) / (1 + p)) ** 2) * (tan(theta)) ** 2)
            a = eps.real
            b = eps.imag
            self.nA1[x] = (sqrt((a + sqrt(a ** 2 + (b ** 2))) / 2)).real
            self.nA1[x] -= 1j * (b / (2 * self.nA1[x]))
        # self.nA1 = 3.4177*ones(self.ln, dtypes=complex)

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

        dela = delay(n1, self.initial_l1)

        ap1 = p1 * tp0 * tp1 * rp2 * exp(2j*pi*f*dela)

        ap2 = 1 - p1 * rp1 * rp2 * exp(2j*pi*f*dela)

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

        dela = delay(n1, self.initial_l1)

        as1 = p1 * ts0 * ts1 * rs2 * exp(2j*pi*f*dela)

        as2 = 1 - p1 * rs1 * rs2 * exp(2j*pi*f*dela)

        h1 = rs0 + as1/as2

        return h1

    def err1(self, n1):    # Returns the error between the theoretical transfer function ratio and extracted transfer
                            # function ratio after changing the complex refractive index at the currently indicated
                            # index to a given complex refractive index
        self.nA1t = copy(self.nA1)
        self.nA1t[self.current1] = n1[0]+1j*n1[1]

        htp = zeros(self.ln, dtype=complex)
        hts = zeros(self.ln, dtype=complex)
        for q in range(0, self.current1 + 1):
            htap = self.h_theory1p(self.FreqS1[q], self.nA1t[q])
            htas = self.h_theory1s(self.FreqS1[q], self.nA1t[q])
            hta_r = float(htap.real)  # /
            hta_i = float(htap.imag)  # numpy problem workaround
            htp[q] = hta_r + 1j * hta_i  # \
            hta_r = float(htas.real)  # /
            hta_i = float(htas.imag)  # numpy problem workaround
            hts[q] = hta_r + 1j * hta_i  # \

        htm = abs(htp[self.current1]) / abs(hts[self.current1])
        htp2 = unwrap(angle(htp))[self.current1] - unwrap(angle(hts))[self.current1]

        mm = self.EXMagS1[self.current1] - htm
        aa = self.EXAngS1[self.current1] - htp2

        er = abs(mm) + abs(aa)
        return er

    def err2(self, n1):    # Returns the error between the theoretical transfer function ratio and extracted transfer
                            # function ratio after changing the complex refractive index at the currently indicated
                            # index to a given complex refractive index
        self.nA1t = copy(self.nA1)
        self.nA1t[self.current1] = n1[0] + 1j * n1[1]

        htp = zeros(self.ln, dtype=complex)
        hts = zeros(self.ln, dtype=complex)
        for q in range(0, self.current1 + 1):
            htap = self.h_theory1p(self.FreqS1[q], self.nA1t[q])
            htas = self.h_theory1s(self.FreqS1[q], self.nA1t[q])
            hta_r = float(htap.real)  # /
            hta_i = float(htap.imag)  # numpy problem workaround
            htp[q] = hta_r + 1j * hta_i  # \
            hta_r = float(htas.real)  # /
            hta_i = float(htas.imag)  # numpy problem workaround
            hts[q] = hta_r + 1j * hta_i  # \

        htm = abs(hts[self.current1]) / abs(htp[self.current1])
        htp2 = unwrap(angle(hts))[self.current1] - unwrap(angle(htp))[self.current1]

        mm = self.EXMagS2[self.current1] - htm
        aa = self.EXAngS2[self.current1] - htp2

        er = abs(mm) + abs(aa)
        return er

    def smoothness1(self, thick):  # Returns the smoothness of the complex refractive functions after changing the
                                    # sample thickness to a given quantity
        nmin = Array('d', range(len(self.FreqS1)))
        kmin = Array('d', range(len(self.FreqS1)))
        lt = Value('d', thick)
        recur2(self.FreqS1, self.es, self.ep, self.esi, self.epi, self.cor_a, self.cor_p, self.cor_a2, self.cor_p2,
               nmin, kmin, lt)

        bb = 1

        nkep = zeros(len(self.FreqS1))
        kkep = zeros(len(self.FreqS1))
        ksi = int(log10(abs(kmin[1])))
        for lm in range(0, len(self.FreqS1)):
            nkep[lm] = round(nmin[lm], 4)
            kkep[lm] = round(-kmin[lm]/(10**ksi), 4)*(10**ksi)

        nkep = lulu(bb, nkep)
        kkep = lulu(bb, kkep)

        nkep = lulu(bb, nkep)
        kkep = lulu(bb, kkep)

        bb = 2
        nkep = lulu(bb, nkep)
        kkep = lulu(bb, kkep)

        for lm in range(bb, len(self.FreqS1) - bb):
            nmin[lm] = nkep[lm]
            kmin[lm] = -kkep[lm]

        self.nA1 = zeros(len(self.FreqS1), dtype=complex)
        for zzk in range(0, len(self.FreqS1)):
            self.nA1[zzk] = nmin[zzk] + 1j*kmin[zzk]
        tt = 0
        for m in range(3, self.ln-3):  # Skip first element due to unstable nature
            de = abs(float(self.nA1[m-1].real)-float(self.nA1[m].real)) \
                 + abs(float(self.nA1[m-1].imag)-float(self.nA1[m].imag))
            tt += de

        return tt

    def call1(self):
        # /\ Use minimisation function to find the thickness of the sample
        # shift = log10(self.initial_l1)
        shift = 10**-5
        # if shift < int(shift):
        #     shift = 10**(int(shift)-2)
        # else:
        #     shift = 10**int(shift-1)
        res = minimize_scalar(self.smoothness1, bounds=(self.initial_l1-shift, self.initial_l1+shift),
                              method='bounded', options={'xatol': 1e-8})
        # res = minimize(self.smoothness1, self.initial_l1, method='nelder-mead',
        #                options={'xtol': 1e-8, 'maxiter': 1000, 'disp': False})
        self.initial_l1 = float(res.x)
        self.initial_l1 = round(self.initial_l1, 7)
        # res = minimize_scalar(self.smoothness1, bounds=(self.initial_l1-shift, self.initial_l1+shift),
        #                       method='bounded', options={'xatol': 1e-8})
        # self.initial_l1 = float(res.x)
        # self.initial_l1 = round(self.initial_l1, 7)
    # \/
        return self.initial_l1

    def call2(self):
        # /\ Use minimisation function to extract the complex refractive index of the sample
        self.initial_generator1(self.EXMagSi, self.EXAngSi)
        # self.nA1 = 3.4177*ones(len(self.EXMagSi), dtype=complex)

        self.current1 = 0

        # for zk in range(0, self.ln):
        #     self.initSet1[0] = self.nA1[zk].real
        #     self.initSet1[1] = self.nA1[zk].imag
        #     res = minimize(self.err1, self.initSet1, method='nelder-mead',
        #                    options={'xtol': 1e-8, 'maxiter': 1000, 'disp': False})
        #     self.nA1[zk] = res.x[0] + 1j*res.x[1]
        #     self.current1 += 1

        # bb = 1
        #
        # nkep = zeros(self.ln)
        # kkep = zeros(self.ln)
        # ksi = int(log10(abs(self.nA1.imag[1])))
        # for lm in range(0, self.ln):
        #     nkep[lm] = round(self.nA1.real[lm], 4)
        #     kkep[lm] = round(-self.nA1.imag[lm] / (10 ** ksi), 4) * (10 ** ksi)
        #
        # nkep = lulu(bb, nkep)
        # kkep = lulu(bb, kkep)
        #
        # nkep = lulu(bb, nkep)
        # kkep = lulu(bb, kkep)
        # bb = 2
        # nkep = lulu(bb, nkep)
        # kkep = lulu(bb, kkep)
        #
        # for lm in range(bb, self.ln - bb):
        #     self.nA1[lm] = nkep[lm]-1j*kkep[lm]
        #
        # self.current1 = 0
        # for zk in range(0, self.ln):
        #     self.initSet1[0] = self.nA1[zk].real
        #     self.initSet1[1] = self.nA1[zk].imag
        #     res = minimize(self.err2, self.initSet1, method='nelder-mead',
        #                    options={'xtol': 1e-8, 'maxiter': 1000, 'disp': False})
        #     self.nA1[zk] = res.x[0] + 1j*res.x[1]
        #     self.current1 += 1

        return self.nA1

    def get_ln(self):
        return self.ln

    def set_len(self, leng):
        self.initial_l1 = leng


def recur(f, S, P, Si, Pi, corA, corP, corA2, corP2, l):
    Work = Worker2()
    Work.set_input(f, S, P, Si, Pi, corA, corP, corA2, corP2)
    Work.set_len(l.value)
    l.value = Work.call1()


def recur2(f, S, P, Si, Pi, corA, corP, corA2, corP2, n, k, l):
    if len(f) >= 64:
        f_low = f[0:int(len(f) / 2)]
        f_high = f[int(len(f) / 2):int(len(f))]
        S_low = S[0:int(len(f) / 2)]
        S_high = S[int(len(f) / 2):int(len(f))]
        P_low = P[0:int(len(f) / 2)]
        P_high = P[int(len(f) / 2):int(len(f))]
        n_low = Array('d', range(int(len(f) / 2)))
        n_high = Array('d', range(int(len(f) / 2)))
        k_low = Array('d', range(int(len(f) / 2)))
        k_high = Array('d', range(int(len(f) / 2)))
        # n_low = n[0:int(len(f) / 2)]
        # n_high = n[int(len(f) / 2):int(len(f))]
        # k_low = k[0:int(len(f) / 2)]
        # k_high = k[int(len(f) / 2):int(len(f))]

        S_lowi = Si[0:int(len(f) / 2)]
        S_highi = Si[int(len(f) / 2):int(len(f))]
        P_lowi = Pi[0:int(len(f) / 2)]
        P_highi = Pi[int(len(f) / 2):int(len(f))]

        corA_low = corA[0:int(len(f) / 2)]
        corA_high = corA[int(len(f) / 2):int(len(f))]
        corP_low = corP[0:int(len(f) / 2)]
        corP_high = corP[int(len(f) / 2):int(len(f))]

        corA2_low = corA2[0:int(len(f) / 2)]
        corA2_high = corA2[int(len(f) / 2):int(len(f))]
        corP2_low = corP2[0:int(len(f) / 2)]
        corP2_high = corP2[int(len(f) / 2):int(len(f))]

        low = Process(target=recur2, args=(f_low, S_low, P_low, S_lowi, P_lowi, corA_low, corP_low,
                                           corA2_low, corP2_low, n_low, k_low, l))
        high = Process(target=recur2, args=(f_high, S_high, P_high, S_highi, P_highi, corA_high,
                                            corP_high, corA2_high, corP2_high, n_high, k_high, l))
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
        Work.set_input(f, S, P, Si, Pi, corA, corP, corA2, corP2)
        Work.set_len(l.value)
        ntemp = Work.call2()
        for zz in range(0, len(f)):
            n[zz] = ntemp[zz].real
            k[zz] = ntemp[zz].imag


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
    Freq = (1 / abs(Time[1] - Time[0])) * (arange(NFFT) - NFFT / 2) / NFFT

    ES = asarray(ES)
    Time = asarray(Time)
    a1 = argmax(abs(ES))
    start0 = where(Time >= Time[a1] - 5.0 * 10 ** -12)[0][0]
    stop0 = where(Time >= Time[a1] + 5.0 * 10 ** -12)[0][0]
    lo = stop0 - start0

    gg = han(a1, len(Time), lo)
    # print(len(gg))
    # gg = zeros(len(Time))
    # print(len(gg))
    # for kkm in range(start0, stop0):
    #     gg[kkm] = 1

    if middy:
        yfp = fft.fftshift(fft.fft(EP, NFFT))
        yfs = fft.fftshift(fft.fft(ES, NFFT))
    else:
        yfp = fft.fftshift(fft.fft(fft.fftshift(EP), NFFT))
        yfs = fft.fftshift(fft.fft(fft.fftshift(ES), NFFT))

    if middy:
        yfpi = fft.fftshift(fft.fft(EP*gg, NFFT))
        yfsi = fft.fftshift(fft.fft(ES*gg, NFFT))
    else:
        yfpi = fft.fftshift(fft.fft(fft.fftshift(EP*gg), NFFT))
        yfsi = fft.fftshift(fft.fft(fft.fftshift(ES*gg), NFFT))

    yc = open('cor.txt', 'r')
    corA = []
    corP = []

    for line in yc:
        temp2 = line.split()
        corA.append(float(temp2[1]))
        corP.append(float(temp2[2]))

    corA = asarray(corA)
    corP = asarray(corP)

    yc = open('cor3.txt', 'r')
    corA2 = []
    corP2 = []

    for line in yc:
        temp2 = line.split()
        corA2.append(float(temp2[1]))
        corP2.append(float(temp2[2]))

    corA2 = asarray(corA2)
    corP2 = asarray(corP2)

    stzo = where(Freq*10**-12 >= botLim)[0][0]
    enzo = stzo + nln
    corA = corA[0:nln]
    corP = corP[0:nln]
    corA2 = corA2[0:nln]
    corP2 = corP2[0:nln]
    freq = Freq[stzo:enzo]
    ess = yfs[stzo:enzo]
    epp = yfp[stzo:enzo]
    essi = yfsi[stzo:enzo]
    eppi = yfpi[stzo:enzo]
    l_ext = Value('d', initial_l)

    # recur(freq, ess, epp, essi, eppi, corA, corP, corA2, corP2, l_ext)
    recur2(freq, ess, epp, essi, eppi, corA, corP, corA2, corP2, next2, kext, l_ext)
    bb = 1

    nkep = zeros(len(freq))
    kkep = zeros(len(freq))
    ksi = int(log10(abs(kext[1])))
    for lm in range(0, len(freq)):
        nkep[lm] = round(next2[lm], 4)
        kkep[lm] = round(-kext[lm]/(10**ksi), 4)*(10**ksi)

    nkep = lulu(bb, nkep)
    kkep = lulu(bb, kkep)

    bb = 1
    nkep = lulu(bb, nkep)
    kkep = lulu(bb, kkep)
    #
    # bb = 2
    # nkep = lulu(bb, nkep)
    # kkep = lulu(bb, kkep)
    for lm in range(bb, len(freq) - bb):
        next2[lm] = nkep[lm]
        kext[lm] = -kkep[lm]

    global freqs, nres, kres
    #
    nres = zeros(nln)
    kres = zeros(nln)
    freqs = zeros(nln)
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
    fr1 = asarray(fr)
    nr1 = asarray(nr)
    ar1 = asarray(ar)
    i1 = where(fr1 >= botLim)[0][0]
    i2 = where(fr1 >= upLim)[0][0]
    fr1 = fr1[i1:i2]
    nr1 = nr1[i1:i2]
    ar1 = ar1[i1:i2]

    mc.m4.clf()
    mc.m4.plot(freqs * 10 ** -12, nres, L='Truncated Bulk Extraction', xL='Frequency (THz)', yL='Refractive Index', CC='r')
    mc.m4.plot(fr1, nr1, L='Input', xl=[botLim, upLim], CC='k')
    # #
    mc.m5.clf()
    mc.m5.plot(freqs * 10 ** -12, kres, L='Truncated Bulk Extraction', xL='Frequency (THz)', yL='Extinction Coefficient', CC='r')
    mc.m5.plot(fr1, ar1, L='Input', xl=[botLim, upLim], CC='k')
    # mc.m5.plot(fr1, ar1 * (c * 100) / (fr1 * (10 ** 12) * 4 * pi), L='Transmission', xl=[botLim, upLim])

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

    yc = open('cor.txt', 'r')

    xt = []
    ypt = []
    yst = []

    corA = []
    corP = []

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

    for line in yc:
        temp2 = line.split()
        corA.append(float(temp2[1]))
        corP.append(float(temp2[2]))

    corA = asarray(corA)
    corP = asarray(corP)
    n_fft = 2 ** (next_pow2(len(xt)))

    time = asarray(xt)
    ypp = asarray(ypt)
    yss = asarray(yst)

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

    # bread = (stop2 - start2) / (2 * sqrt(2 * log(2)))
    # gg = gauss(linspace(0, len(time), len(time)), b=argmax(abs(yss)), ccd=bread)

    freq = (1 / abs(time[1] - time[0])) * (arange(n_fft) - n_fft / 2) / n_fft

    if middy:
        # y_p_freq = fft.fftshift(fft.fft(ypp*gg, n_fft))
        # y_s_freq = fft.fftshift(fft.fft(yss*gg, n_fft))
        y_p_freq = fft.fftshift(fft.fft(ypp, n_fft))
        y_s_freq = fft.fftshift(fft.fft(yss, n_fft))
    else:
        # y_p_freq = fft.fftshift(fft.fft(fft.fftshift(ypp*gg), n_fft))
        # y_s_freq = fft.fftshift(fft.fft(fft.fftshift(yss*gg), n_fft))
        y_p_freq = fft.fftshift(fft.fft(fft.fftshift(ypp), n_fft))
        y_s_freq = fft.fftshift(fft.fft(fft.fftshift(yss), n_fft))

    # ff, aa, pp = correction()
    # start = 0

    # for i in range(int(n_fft/2), n_fft):
    #     if freq[i] >= ff[0]:
    #         start = i
    #         break

    # y_p_freq2 = y_p_freq[start: len(ff)+start]
    # y_p_freq2a = abs(y_p_freq2)
    # y_p_freq2p = unwrap(angle(y_p_freq2))
    #
    # for i in range(start, len(ff)+start):
    #     y_p_freq[i] = aa[i-start]*y_p_freq2a[i-start]*exp(1j*(y_p_freq2p[i-start] + pp[i-start]))

    h_experiment = zeros(n_fft, dtype=complex)

    for i in range(0, n_fft):
        h_experiment[i] = y_p_freq[i] / y_s_freq[i]

    # startzone = 0
    # zone = 0
    # for i in range(0, n_fft):
    #     if freq[i] == 0:
    #         startzone = i
    #         break
    # for i in range(startzone, n_fft):
    #     zone += 1
    #     if freq[i] >= upLim * 10 ** 12:
    #         break
    # endz = startzone + zone
    # startzone2 = 0
    # for i in range(startzone, endz):
    #     if freq[i] >= botLim * 10 ** 12:
    #         startzone2 = i
    #         break

    startzone2 = where(freq >= botLim * 10 ** 12)[0][0]
    endz = where(freq >= upLim * 10 ** 12)[0][0]

    # endz = startzone + zone
    # cuts = endz - startzone2
    # t_end = startzone2 + cuts
    freq_s = freq[startzone2:endz]
    exh = h_experiment[startzone2:endz]
    ln = len(exh)

    global freqs, nres, kres
    freqs = copy(freq_s)
    nres = zeros(ln, dtype=complex)
    kres = zeros(ln, dtype=complex)

    # print(len(corP), len(corA), len(exh))

    ex_ang = unwrap(angle(exh)) #- corP
    ex_mag = abs(exh)#/corA

    for x in range(0, ln):
        p = ex_mag[x] * (e ** (-1j * (ex_ang[x])))
        eps = (n0 ** 2) * ((sin(theta)) ** 2) * (1 + (((1 - p) / (1 + p)) ** 2) * (tan(theta)) ** 2)
        a = eps.real
        b = eps.imag
        nres[x] = (sqrt((a + sqrt(a ** 2 + (b ** 2))) / 2)).real
        kres[x] = b / (2 * nres[x])

    # nn = open('GlassRef.txt', 'r')
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
    fr1 = asarray(fr)
    nr1 = asarray(nr)
    ar1 = asarray(ar)
    i1 = where(fr1 >= botLim)[0][0]
    i2 = where(fr1 >= upLim)[0][0]
    fr1 = fr1[i1:i2]
    nr1 = nr1[i1:i2]
    ar1 = ar1[i1:i2]

    # fr1 = asarray(fr)*10**12
    # nr1 = asarray(nr)
    # ar1 = asarray(ar)

    # ner = zeros(len(nres))
    # ker = zeros(len(kres))

    # start = 0
    # for i in range(0, len(ner)):
    #     for j in range(start, len(fr1)):
    #         if fr1[j] >= freqs[i]:
    #             start = j
    #             ner[i] = 100 * abs(nres[i] - nr1[j]) / nr1[j]
    #             ker[i] = 100 * abs(kres[i] - ar1[j]) / ar1[j]
    #             break

    # for mnn in range(0, len(kres)):
    #     if kres[mnn] < 0:
    #         kres[mnn] = 0
    #         nres[mnn] = nres[mnn-1]

    mc.m1.clf()
    mc.m1.plot(time * 10 ** 12, yss, L='ES', xL='Time (ps)', yL='Amplitude(a.u.)')
    mc.m1.plot(time * 10 ** 12, ypp, L='EP')
    mc.m4.clf()
    mc.m4.plot(freq_s * 10 ** -12, nres, L='Ellipsometry', xL='Frequency (THz)', yL='Refractive Index', CC='r')
    mc.m4.plot(fr1, nr1, L='Input', xl=[botLim, upLim], CC= 'k')
    # mc.m4.plot(freqs * 10 ** -12, ner, L='Refractive Index', xL='Frequency (THz)', yL='Error(%)', xl=[botLim, upLim])
    #
    mc.m5.clf()
    mc.m5.plot(freq_s * 10 ** -12, kres, L='Ellipsometry', xL='Frequency (THz)', yL='Extinction Coefficient', CC='r')
    # mc.m5.plot(fr1, ar1 * (c * 100) / (fr1 * (10 ** 12) * 4 * pi), L='Transmission', xl=[botLim, upLim])
    mc.m5.plot(fr1, ar1, L='Input', xl=[botLim, upLim], CC='k')
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
    yc = open('cor3.txt', 'r')
    corA = []
    corP = []

    n_sil = 3.4177

    for line in yc:
        temp2 = line.split()
        corA.append(float(temp2[1]))
        corP.append(float(temp2[2]))

    corA = asarray(corA)
    corP = asarray(corP)

    fp = open(file_EP, 'r')
    fs = open(file_ES, 'r')

    t = []
    EP = []
    ES = []

    for line in fp:
        temp = line.replace(',', '.').split()
        t.append(float(temp[0]) * 10 ** -12)
        EP.append(float(temp[1]))

    for line in fs:
        temp = line.replace(',', '.').split()
        ES.append(float(temp[1]))

    Ep = asarray(EP)
    Es = asarray(ES)
    t = asarray(t)

    a1 = argmax(abs(Es))
    start0 = where(t >= t[a1] - 2.5 * 10 ** -12)[0][0]
    stop0 = where(t >= t[a1] + 2.5 * 10 ** -12)[0][0]
    lo = stop0 - start0

    gg = han(argmax(abs(Es)), len(t), lo)
    gg2 = han(argmax(abs(Es - Es * gg)), len(t), lo)

    gg3 = han(argmax(abs(Ep)), len(t), lo)
    gg4 = han(argmax(abs(Ep - Ep * gg3)), len(t), lo)

    Ep_ref = Ep * gg3
    Ep_sam = Ep * gg4

    Es_ref = Es * gg
    Es_sam = Es * gg2

    nfft = 2 ** next_pow2(len(Ep))

    Ep_ref_f = fft.fftshift(fft.fft(Ep_ref, nfft))
    Ep_sam_f = fft.fftshift(fft.fft(Ep_sam, nfft))

    Es_ref_f = fft.fftshift(fft.fft(Es_ref, nfft))
    Es_sam_f = fft.fftshift(fft.fft(Es_sam, nfft))

    freq = (1 / abs(t[1] - t[0])) * (arange(nfft) - nfft / 2) / nfft

    start = where(freq >= botLim * 10 ** 12)[0][0]
    stop = where(freq >= upLim * 10 ** 12)[0][0]

    tra_p = tp12(n0, n_sil) * tp12(n_sil, n0)
    tra_s = ts12(n0, n_sil) * ts12(n_sil, n0)

    freq = freq[start:stop]

    Ep_ref_f = Ep_ref_f[start:stop]
    Ep_sam_f = Ep_sam_f[start:stop] / tra_p

    Es_ref_f = Es_ref_f[start:stop]
    Es_sam_f = Es_sam_f[start:stop] / tra_s

    Ap_ref = abs(Ep_ref_f)/corA
    Ap_sam = abs(Ep_sam_f)/corA

    As_ref = abs(Es_ref_f)/corA
    As_sam = abs(Es_sam_f)/corA

    Pp_ref = unwrap(angle(Ep_ref_f))-corP
    Pp_sam = unwrap(angle(Ep_sam_f))-corP

    Ps_ref = unwrap(angle(Es_ref_f))-corP
    Ps_sam = unwrap(angle(Es_sam_f))-corP

    err_a = (Ap_ref / As_ref) / (-rp12(n0, n_sil) / rs12(n0, n_sil))
    err_p = pi - (Pp_ref - Ps_ref)

    p = (Ap_sam / As_sam) * exp(-1j * (Pp_sam - Ps_sam + err_p)) / err_a
    theta1 = arcsin(n0 * sin(theta) / n_sil)

    eps = (n_sil ** 2) * ((sin(theta1)) ** 2) * (1 + (((1 - p) / (1 + p)) ** 2) * ((tan(theta1)) ** 2))

    Re = eps.real
    Im = eps.imag

    n_out = sqrt((Re + sqrt(Re ** 2 + Im ** 2)) / 2)
    k_out = Im / (2 * n_out)

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
    fr22 = asarray(fr2)
    nr22 = asarray(nr2)
    kr22 = asarray(ar2)
    i1 = where(fr22 >= botLim)[0][0]
    i2 = where(fr22 >= upLim)[0][0]
    fr22 = fr22[i1:i2] * 10 ** 12
    nr22 = nr22[i1:i2]
    # ar1 = asarray(ar)
    kr22 = kr22[i1:i2]

    # fr22 = asarray(fr2) * 10 ** 12
    # nr22 = asarray(nr2)
    # # ar1 = asarray(ar)
    # kr22 = asarray(ar2)
    #
    # ner = zeros(len(nres2))
    # ker = zeros(len(kres2))
    #
    # start = 0
    # for i in range(0, len(ner)):
    #     for j in range(start, len(fr2)):
    #         if fr22[j] >= freq_s[i]:
    #             start = j
    #             ner[i] = 100 * abs(nres2[i] - nr22[j]) / nr22[j]
    #             ker[i] = 100 * abs(kres2[i] - kr22[j]) / kr22[j]
    #             break

    mc.m1.clf()
    # mc.m1.plot(time * 10 ** 12, yss*gg, L='ES', xL='Time (ps)', yL='Amplitude(a.u.)')
    # mc.m1.plot(time * 10 ** 12, ypp*gg, L='EP')
    mc.m1.plot(t * 10 ** 12, Es, L='ES', xL='Time (ps)', yL='Amplitude(a.u.)')
    mc.m1.plot(t * 10 ** 12, Ep, L='EP')
    # mc.m1.plot(time * 10 ** 12, yss*gg2, L='ES2')
    # mc.m1.plot(time * 10 ** 12, ypp*gg2, L='EP2')
    # mc.m1.plot(time * 10 ** 12, yss, L='ES0')
    # mc.m1.plot(time * 10 ** 12, ypp, L='EP0')
    # mc.m1.semilogy(freq_s * 10 ** -12, abs(y_p_freq2n), L='Ellipsometry',
    #  xL='Frequency (THz)', yL='Refractive Index')
    # mc.m1.semilogy(freq_s * 10 ** -12, abs(y_s_freq2n), L='Transmission', xl=[botLim, upLim])
    # mc.m1.plot(freq_s * 10 ** -12, unwrap(angle(y_p_freq2n)), L='Ellipsometry',
    # xL='Frequency (THz)', yL='Refractive Index')
    # mc.m1.plot(ff * 10 ** -12, aa, L='Ellipsometry', xL='Frequency (THz)',
    #            yL='Refractive Index')
    # mc.m1.plot(freq_s * 10 ** -12, unwrap(angle(y_s_freq2n)), L='Transmission', xl=[botLim, upLim])
    mc.m4.clf()
    # mc.m4.plot(freq_s * 10 ** -12, nres1, L='Layer1', xL='Frequency (THz)', yL='Refractive Index')
    # mc.m4.plot(freq_s * 10 ** -12, nres2, L='Layer2', xl=[botLim, upLim])
    mc.m4.plot(freq * 10 ** -12, n_out, L='Extracted Layer 2', xL='Frequency (THz)', yL='Refractive Index', CC='r')
    mc.m4.plot(fr22 * 10 ** -12, nr22, L='Input', xl=[botLim, upLim], CC='k')
    # mc.m4.plot(freq_s * 10 ** -12, ner, L='Refractive Index', xL='Frequency (THz)', yL='Error(%)', xl=[botLim, upLim])
    #
    mc.m5.clf()
    # mc.m5.plot(freq_s * 10 ** -12, kres1, L='Layer1', xL='Frequency (THz)', yL='Extinction Coefficient')
    # mc.m5.plot(freq_s * 10 ** -12, kres2, L='Layer2', xl=[botLim, upLim])
    mc.m5.plot(freq * 10 ** -12, k_out, L='Extracted Layer 2', xL='Frequency (THz)', yL='Extinction Coefficient', CC='r')
    mc.m5.plot(fr22 * 10 ** -12, kr22, L='Input', xl=[botLim, upLim], CC='k')
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

        else:
            self.ref.show()
            self.btn7.show()
            if self.ret == 's':
                self.lbl2.hide()
                self.thick.hide()
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

                pha_p = unwrap(angle(pf_plot))
                pha_s = unwrap(angle(sf_plot))

                self.m1.clf()
                self.m1.plot(time_h * 10 ** 12, s_h, L='ES', xL='Time(ps)', yL='Amplitude(a.u.)')
                self.m1.plot(time_h * 10 ** 12, p_h, L='EP')

                self.m2.clf()
                self.m2.semilogy(freq_plot * 10 ** -12, abs(sf_plot), L='ES', xL='Frequency (THz)',
                                 yL='Amplitude(a.u.)')
                self.m2.semilogy(freq_plot * 10 ** -12, abs(pf_plot), L='EP', xl=[botLim, upLim])
                #
                self.m3.clf()
                self.m3.plot(freq_plot * 10 ** -12, pha_s, L='ES', xL='Frequency (THz)', yL='Phase(Rads)')
                self.m3.plot(freq_plot * 10 ** -12, pha_p, L='EP', xl=[botLim, upLim])

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

                pha_p = unwrap(angle(pf_plot))
                pha_s = unwrap(angle(sf_plot))

                self.m1.clf()
                self.m1.plot(time_h * 10 ** 12, s_h, L='ES', xL='Time(ps)', yL='Amplitude(a.u.)')
                self.m1.plot(time_h * 10 ** 12, p_h, L='EP')

                self.m2.clf()
                self.m2.semilogy(freq_plot * 10 ** -12, abs(sf_plot), L='ES', xL='Frequency (THz)',
                                 yL='Amplitude(a.u.)')
                self.m2.semilogy(freq_plot * 10 ** -12, abs(pf_plot), L='EP', xl=[botLim, upLim])
                #

                self.m3.clf()
                self.m3.plot(freq_plot * 10 ** -12, pha_s, L='ES', xL='Frequency (THz)', yL='Phase(Rads)')
                self.m3.plot(freq_plot * 10 ** -12, pha_p, L='EP', xl=[botLim, upLim])
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

                elif self.ret == 'b':
                    self.lbl2.setStyleSheet('QLabel {color: black;}')
                    self.btn2.hide()
                    self.lbl1.show()
                    start_b(self)
                    # thread = threading.Thread(target=start_b, args=(self,))
                    # thread.daemon = True
                    # thread.start()

                elif self.ret == 'd':
                    self.lbl2.setStyleSheet('QLabel {color: black;}')
                    self.btn2.hide()
                    self.lbl1.show()
                    start_d2(self)
                    # thread = threading.Thread(target=start_d2, args=(self,))
                    # thread.daemon = True
                    # thread.start()


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, x, y, L=None, xL=None, yL=None, xl=None, yl=None, CC=None):
        ax = self.figure.add_subplot(111)
        if not L:
            ax.plot(x, y)
        else:
            ax.plot(x, y, label=L, color=CC)
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
