from cmath import *
import numpy as np
import math
import matplotlib.pyplot as pl
import random
theta = 60.73*pi/180  # #/initial angle of incidence
c = 299792458  # #/speed of light in m/s0
n_i = 1.00027+0j  # #/Initial refractive index


def theta_trans(n_out):   # /Generate transmitted angle\
    return asin((n_i/n_out)*sin(theta))


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


def abso(k, freq):  # #/convert extinction coefficient to absorption coefficient
    return 2*pi*freq*k/(c*100)


def kgen(alpha, freq):  # #/convert absorption coefficient to extinction coefficient
    return 100*alpha*c/(2*pi*freq)


def nextpow2(a):
    m = log(a)/log(2)
    a = int(m.real)
    if a < m.real:
        return a+1
    else:
        return a


def currentj(dt):  # #/Current in antenna
    t = dt
    aa = 100 * 10**-15
    bb = 200 * 10**-15
    cc = 25 * 10**-15
    intea = (1/(2*math.sqrt(aa**-2)))*math.sqrt(pi)*math.e**(-(t**2)/(aa**2))
    inteb1 = (-math.e**(((aa**2 - 2*bb*t)**2)/(4 * aa**2 * bb**2)))
    inteb2 = (-1 + math.erf((math.sqrt(aa**-2)*(aa**2 - 2*bb*t))/(2*bb)))
    intec1 = (-math.e**((aa**2 * (bb+cc) - 2*bb*cc*t)**2 / (4 * aa**2 * bb**2 * cc**2)))
    intec2 = (-1 + math.erf((math.sqrt(aa**-2)*(aa**2 * (bb + cc) - 2*bb*cc*t))/(2*bb*cc)))
    inteb = inteb1*inteb2
    intec = intec1*intec2
    intecb = (inteb + intec)
    # print(t,InteA,InteCB)
    inte = intea*intecb
    charge_e = 1.60217662 * 10**-19
    mass_e = 9.10938356 * 10**-31
    return (charge_e*cc/mass_e)*inte


def elecgen(t):  # #/Generated electric field from antenna
    cc = 25*10**-15
    l2 = 10*10**-6
    diff = 10*10**-22
    charge_e = 1.60217662 * 10**-19
    mass_e = 9.10938356 * 10**-31
    r = 35*10**-2
    epsi = 8.854187817*10**-12
    current_diff = (currentj(t+diff)-currentj(t))/diff
    return 200*10**15 * (l2/(4*math.pi*epsi*c*c*r))*(charge_e*cc/mass_e * current_diff)


def hh(x):
    if x == 0:
        return 1
    else:
        return (1+(x/abs(x)))/2


def resonance(freqy, freqsy):
    ll = len(freqsy)
    nr = np.zeros(ll)
    absr = np.zeros(ll)
    p = int(freqy/(freqsy[1]-freqsy[0]))
    for k in range(0, ll):
        zzk = (k - p - ll/2)/16
        # if(i==0):
        #    nr[k] = 0
        # elif(i<=pi and i>=-pi):
        #    nr[k] = sin(i).real
        # else:
        #    nr[k] = 0
        if freqy < 0:
            nr[k] = -0.5*zzk*e**(-zzk**2/16)
        else:
            nr[k] = 0.5*zzk*e**(-zzk**2/16)
        absr[k] = (10/pi)*(2/(zzk**2 + 4))
    return nr, absr


def resonance2(freqy, freqsy):
    ll = len(freqsy)
    nr = np.zeros(ll)
    absr = np.zeros(ll)
    p = int(freqy/(freqsy[1]-freqsy[0]))
    for k in range(0, ll):
        zzk = (k - p)/16
        # if(i==0):
        #    nr[k] = 0
        # elif(i<=pi and i>=-pi):
        #    nr[k] = sin(i).real
        # else:
        #    nr[k] = 0
        if freqy < 0:
            nr[k] = -0.5*zzk*e**(-zzk**2/16)
        else:
            nr[k] = 0.5*zzk*e**(-zzk**2/16)
        absr[k] = (10/pi)*(2/(zzk**2 + 4))
    return nr, absr


d1 = 0.1  # #\thickness of layer 1 cm

d2 = 2.0  # #\thickness of layer 2

tstep = 1.0*10**-15  # #\temporal distance between 2 measured points in data
time = []
ts22 = 0
while ts22 <= 200*10**-12:
    time.append(ts22)
    ts22 += tstep
NFFT = 2*2**(nextpow2(len(time)))

E0 = np.zeros(NFFT)


FS = 1/tstep
freqs = FS*np.linspace(0, 1, NFFT)

time2 = (1/freqs[1])*(np.arange(NFFT)-NFFT/2)/NFFT
Freqs2 = (1/abs(time2[1]-time2[0]))*(np.arange(NFFT)-NFFT/2)/NFFT
for i in range(0, NFFT):
    if (time2[i] >= -2*10**-12) and (time2[i] <= 2*10**-12):
        E0[i] = elecgen(time2[i])

E0F = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E0), NFFT))
Ph = np.unwrap(np.angle(E0F))

E0F2 = np.abs(E0F)*np.exp(1j*Ph)

fres = np.array([1.2, 1.3, 1.6, 1.65, 2.2, 2.25, 2.3, 2.5, 2.6, 2.7, 2.95, 3.0, 3.2, 3.6]) * 10**12
fres2 = 0.0-fres
alp = np.zeros(NFFT)
alpt = np.zeros(NFFT)
ntt = np.zeros(NFFT)
A = (1.6**2 * 6.0221*10**12)/(2*8.85*9.11) * 10**5
Er1 = 6.0221*10**13
Gr1 = 10**9
nt = 2*np.ones(NFFT)
for St in range(0, len(fres)):
    ntt, alpt = resonance(fres[St], Freqs2)
    for i in range(0, NFFT):
        nt[i] += 0.05*ntt[i]
        alp[i] += 0.05*alpt[i]
for St in range(0, len(fres2)):
    ntt, alpt = resonance(fres2[St], Freqs2)
    for i in range(0, NFFT):
        nt[i] += 0.05*ntt[i]
        alp[i] += 0.05*alpt[i]

n1 = np.zeros(NFFT, dtype=complex)
n2 = np.zeros(NFFT, dtype=complex)
rp1 = np.zeros(NFFT, dtype=complex)
tp1 = np.zeros(NFFT, dtype=complex)
tp21 = np.zeros(NFFT, dtype=complex)
rp2 = np.zeros(NFFT, dtype=complex)
rs1 = np.zeros(NFFT, dtype=complex)
ts1 = np.zeros(NFFT, dtype=complex)
ts21 = np.zeros(NFFT, dtype=complex)
rs2 = np.zeros(NFFT, dtype=complex)
v1 = np.zeros(NFFT)
d1f = np.zeros(NFFT)
travt = np.zeros(NFFT)
travts = np.zeros(NFFT)
theta1 = np.zeros(NFFT, dtype=complex)
theta2 = np.zeros(NFFT, dtype=complex)
for i in range(0, NFFT):
    if Freqs2[i] == 0:
        n1[i] = 3.25 - 1j*kgen(4.0, Freqs2[i+1])
        # n2[i] = nt[i] - 1j*kgen(alp[i+1], Freqs2[i+1])
        n2[i] = 2.0 - 1j*kgen(200.0, Freqs2[i+1])
    else:
        n1[i] = 3.25 - 1j*kgen(4.0, Freqs2[i])
        # n2[i] = nt[i] - 1j*kgen(alp[i], Freqs2[i])
        n2[i] = 2.0 - 1j*kgen(200.0, Freqs2[i])

for i in range(0, NFFT):
    theta1[i] = theta_trans(n1[i])
    theta2[i] = theta_trans(n2[i])
    d1f[i] = d1/cos(theta1[i].real).real
    v1[i] = c/n1[i].real
    travt[i] = 2*(d1f[i].real/100)/v1[i].real
# print(travt)
# tempt = np.min(travt)
# tshiftMin = (time2[1]-time2[0])*int(tempt/(time2[1]-time2[0]))
# tshiftMins = int(tempt/(time2[1]-time2[0]))
# nnew = np.zeros(NFFT, dtype=complex)
# dnew = np.zeros(NFFT)
extinct = np.zeros(NFFT, dtype=complex)
# if (tempt-tshiftMin)/(time2[1]-time2[0]) >= 0.5:
#     tshiftMin += (time2[1]-time2[0])
#     tshiftMins += 1
#
# for i in range(0, NFFT):
#     if np.isnan(((travt[i]-tempt)/(time2[1]-time2[0]))):
#         temp = 0
#     else:
#         temp = ((travt[i] - tempt) / (time2[1] - time2[0]))
#         # if time2[i] == 0:
#         #     temp = 0
#         # else:
#         #     temp = ((travt[i]-tempt)/(time2[1]-time2[0]))
#     if temp - int(temp) >= 0.5:
#         travts[i] = tshiftMins + int(temp)+1
#         travt[i] = tshiftMin + (time2[1]-time2[0])*(int(temp)+1)
#     else:
#         travts[i] = tshiftMins + int(temp)
#         travt[i] = tshiftMin + (time2[1]-time2[0])*(int(temp))
#     # dnew[i] = travt[i] * v1[i]
#     n1[i] = sqrt(travt[i]*c/(d1/100) - n_i * n_i * sin(theta) * sin(theta)) + 1j*n1[i].imag
for i in range(0, NFFT):
    if np.isnan(travt[i] / (time2[1]-time2[0])):
        temp = 0
    else:
        temp = travt[i] / (time2[1] - time2[0])
    if temp - int(temp) >= 0.5:
        travts[i] = int(temp)+1
        travt[i] = (time2[1]-time2[0])*(int(temp)+1)
    else:
        travts[i] = int(temp)
        travt[i] = (time2[1]-time2[0])*(int(temp))
    # dnew[i] = travt[i] * v1[i]
    n1[i] = sqrt(((travt[i]*c)**2)/((2*d1/100)**2) - n_i * n_i * sin(theta) * sin(theta)) + 1j*n1[i].imag
# print(travt)
##

for i in range(0, NFFT):
    theta1[i] = theta_trans(n1[i])
    theta2[i] = theta_trans(n2[i])
    rp1[i] = rp12(n_i, n1[i])
    tp1[i] = tp12(n_i, n1[i])
    tp21[i] = tp12(n1[i], n_i)
    rp2[i] = rp12(n1[i], n2[i])
    rs1[i] = rs12(n_i, n1[i])
    ts1[i] = ts12(n_i, n1[i])
    ts21[i] = ts12(n1[i], n_i)
    rs2[i] = rs12(n1[i], n2[i])
    d1f[i] = 0.02*d1/cos(theta1[i].real).real
    # v1[i] = c/n1[i].real
    # travt[i] = 2*(d1f[i].real/100)/v1[i].real
# #  Material Interaction

count = 1
E0F3s = np.zeros(NFFT, dtype=complex)
E0F3p = np.zeros(NFFT, dtype=complex)
for i in range(0, NFFT):
    # Shift = (-2j*pi*(i-NFFT/2)*travts[i]/NFFT)
    extinct[i] = -2j * pi * Freqs2[i] * n1[i] * d1f[i] / c
    A = e**(extinct[i])
    E0F3s[i] = E0F2[i]*(rs1[i] + ts1[i]*ts21[i]*rs2[i]*A/(1.0 - A*rs1[i]*rs2[i]))
    E0F3p[i] = E0F2[i]*(rp1[i] + tp1[i]*tp21[i]*rp2[i]*A/(1.0 - A*rp1[i]*rp2[i]))

# #  Measured Values
E0s = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E0F3s), NFFT)).real
E0p = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E0F3p), NFFT)).real
E000 = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E0F2), NFFT)).real
start = 0
stop = 0
for i in range(0, NFFT):
    if time2[i] >= -10*10**-12:
        start = i
        break
for i in range(start, NFFT):
    if time2[i] >= 65*10**-12:
        stop = i
        break
# pl.plot(time2, -E0s/(np.max(np.abs(E0s))))
# pl.plot(time2, E0p/(np.max(np.abs(E0p))))
# pl.plot(time2, E000/(np.max(np.abs(E000))))
time2 = time2[start:stop]
E0s = E0s[start:stop]
E0p = E0p[start:stop]
E0002 = E000[start:stop]
nn0 = int((10*10**-15)/(time2[1] - time2[0]))
if ((10*10**-15)/(time2[1] - time2[0])) - nn0 >= 0.5:
    nn0 += 1
E0s2 = np.zeros(int(len(time2)/nn0))
E02 = np.zeros(int(len(time2)/nn0))
E0p2 = np.zeros(int(len(time2)/nn0))
time22 = np.zeros(int(len(time2)/nn0))


for i in range(0, int(len(time2)/nn0)):
    E0s2[i] = E0s[i * nn0]
    E0p2[i] = E0p[i * nn0]
    E02[i] = E0002[i * nn0]
    time22[i] = time2[i * nn0]

pl.plot(time22, E02)
pl.plot(time2, E0s)
pl.plot(time22, E0s2)
pl.show()
pl.plot(time22, E02)
pl.plot(time2, E0p)
pl.plot(time22, E0p2)
# pl.plot(time2, E000)
pl.show()

YS = open('ES.txt', 'w')
YP = open('EP.txt', 'w')
YN = open('N1Ref.txt', 'w')
YN2 = open('N2Ref.txt', 'w')

# for i in range(0, stop - start):
#     SS = (str((time2[i])*10**12)+"\t"+str(E0s[i].real)+"\t"+str(dnew[i]/2 * cos(theta1[i].real).real)+"\n")
#     YS.write(SS)
#     SS = (str((time2[i])*10**12) + "\t" + str(E0p[i].real) + "\t"+"\n")
#     YP.write(SS)
for i in range(0, len(time22)):
    SS = (str((time22[i])*10**12)+"\t"+str(E0s2[i].real)+"\t"+"\n")
    YS.write(SS)
    SS = (str((time22[i])*10**12) + "\t" + str(E0p2[i].real) + "\t"+"\n")
    YP.write(SS)
for i in range(0, NFFT):
    SS = (str((Freqs2[i]) * 10 ** -12) + "\t" + str(n1[i].real) + "\t" + str(-n1[i].imag) + "\n")
    YN.write(SS)
    SS = (str((Freqs2[i]) * 10 ** -12) + "\t" + str(n2[i].real) + "\t" + str(-n2[i].imag) + "\n")
    YN2.write(SS)
YS.close()
YP.close()
YN.close()
YN2.close()

