from cmath import *
import numpy as np
import math
import matplotlib.pyplot as pl

theta = 0  # #/initial angle of incidence
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


def kgen(alpha, freq):  # #/convert absorption coefficient to extinction coefficient
    return 100*alpha*c/(2*pi*freq)


n = 3.125 - 0.03j

size = 90000

thet = np.zeros(size)
rs = np.zeros(size, dtype=complex)
rp = np.zeros(size, dtype=complex)
ze = np.zeros(size)

for i in range(0, size):
    thet[i] = i/1000
    theta = thet[i]*pi/180
    rs[i] = rs12(n_i, n)
    rp[i] = rp12(n_i, n)

pl.plot(thet, rs, label='rs')
pl.plot(thet, rp, label='rp')
pl.plot(thet, ze, 'k--')
pl.legend()
pl.xlabel('Angle of incidence (degrees)')
pl.ylabel('Reflection coefficient')
pl.show()