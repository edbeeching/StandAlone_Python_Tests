# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 08:18:11 2017

@author: Edward

This program implements the Fourier Transform



"""
#================================ IMPORTS =====================================

import numpy as np
import matplotlib.pyplot as plt  

#==============================================================================


def discrete_fourier_transform(data):
    num_samples = len(data)
    
    result = np.zeros((num_samples,), dtype='complex')
    
    for k in range(num_samples):
        for jj in range(num_samples):
            result[k] += data[jj] * np.exp( -1j * 2.0 * np.pi * k * jj / num_samples)
        
    return result












if __name__ == '__main__':
    
    # First generate some data and plot it
    # signal will have 100 Hz sampling frequency and be 2 seconds long (201 samples)
    time = np.linspace(0.0, 2.0, num=201)
    # 5 Hz sin wave
    signal1 = np.sin(5*time*2*np.pi)
    # 2 Hz sin wave
    signal2 = np.sin(2*time*2*np.pi)
    plt.plot(time, signal1+signal2)
    
    tt = np.zeros((1,1), dtype='complex')
    tt[0] = 1 + 2j
    
    signal3 = signal1+signal2
    
    res = discrete_fourier_transform(signal1+signal2)

    real = np.abs(res)
    
    fs = 100
    
    freqs = np.linspace(0,fs/2, 101)
    

    from numpy.fft import rfft

    res2 = rfft(signal1+signal2)
    real2 = np.abs(res2)  
    
    
    plt.semilogy(freqs, real[:101])
    plt.semilogy(freqs, real2)
