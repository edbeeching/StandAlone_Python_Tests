# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:56:50 2017

@author: Edward
"""

import numpy as np
from numpy.fft import fft2
from scipy import misc
import matplotlib.pyplot as plt

lena = misc.ascent().astype(np.float32)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(lena,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(np.roll(np.roll(np.log10(np.abs(fft2(lena))),256, axis=1),256, axis=0))
plt.subplot(1,3,3)
plt.imshow(np.roll(np.roll(np.log10(np.abs(fft2(lena[:,::2]))),128, axis=1),256, axis=0))