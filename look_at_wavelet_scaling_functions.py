# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:02:07 2017

@author: Edward
"""

import matplotlib.pyplot as plt

from pywt import Wavelet
from numpy import convolve
from math import sqrt
import numpy as np


def next_scale(current_scale, h):
    out = np.zeros((len(current_scale)*2 -1,))
    out[::2] = current_scale
    return convolve(out, h)*sqrt(2)

def pad(g):
    return np.lib.pad(g,(1,2),'constant', constant_values=(0,0))




wavelet = Wavelet('db2')
phi, psi, x = wavelet.wavefun(level=1)
low_rec = wavelet.rec_lo
scaling_function = sqrt(2.0)*np.array(low_rec)

plt.figure()
plt.subplot(2,3,1)
plt.title('Scaling function for {}'.format(1))
plt.plot(phi)
plt.plot(pad(scaling_function), 'o',c='r')
for i, level in enumerate(range(2,7),2):
    plt.subplot(2, 3, i)
    plt.title('Scaling function for {}'.format(level))
    
    phi, psi, x = wavelet.wavefun(level=level)
    scaling_function = next_scale(scaling_function, low_rec)
    plt.plot(phi)
    plt.plot(pad(scaling_function), 'o', c='r')



import math
h = [1.0/math.sqrt(2), 0.0,0.0, 1.0/math.sqrt(2)]

plt.title('Scaling function for {}'.format(1))






