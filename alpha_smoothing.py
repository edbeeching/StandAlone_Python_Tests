# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:17:34 2017

@author: Edward
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,1,200)
y = np.sin(2*np.pi*x) + 0.1*np.random.randn(x.shape[0])

plt.suptitle('Plot of Alpha smoothing of function y = sin(x) + N(0, 0.1), using sm(t+1) = alpha*sm(t) + (1-alpha)*y')
plt.subplot(1,2,1)
plt.plot(x,y, label='Unsmoothed Function', marker='o')
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.subplot(1,2,2)
plt.semilogy(np.abs(np.fft.rfft(y)), label='Unsmoothed Function', marker='o')
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
plt.minorticks_on()
for alpha in [0.0, 0.3,0.7,0.9,1.0]:
    m = 0
    y_filt = []
    for yy in y:
        m = alpha*m + (1-alpha)*yy
        y_filt.append(m) 
    plt.subplot(1,2,1)
    plt.title('Alpha smoothing')
    plt.plot(x, np.array(y_filt), label='Alpha={}'.format(alpha))
    plt.subplot(1,2,2)
    plt.title('FFT of smoothing')
    plt.semilogy(np.abs(np.fft.rfft(y_filt)), label='Alpha={}'.format(alpha))
    
    
plt.legend()
plt.show()

