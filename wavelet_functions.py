# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:28:25 2017

@author: Edward
"""

import numpy as np
import matplotlib.pyplot as plt


def haar(t):
    if 0.0 <= t <=1.0:
        return 1.0
    else:
        return 0.0
vhaar = np.vectorize(haar)    

def hat(t):
    if 0.0 <= t <= 1.0:
        return t
    elif 1.0<=t<=2.0:
        return 2.0-t
    else: return 0.0
vhat = np.vectorize(hat)
 
def spline(t):
    if 0.0 <= t <= 1.0:
        return t**2.0
    elif 1.0 <- t <=2.0:
        return -2*(t**2.0) + (6.0*t) -3.0
    elif 2.0 <= t <= 3.0:
        return (3.0-t)**2.0
    else:
        return 0
vspline = np.vectorize(spline)
x = np.linspace(-4.0,4.0,1001)
y = vspline(x)

plt.plot(x,y)

        
