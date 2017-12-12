# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:19:48 2017

@author: Edward
"""

# Program to show exaples of regularisation when applying regression to a 
# polynomial of order 7

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-0.9,0.9,11)

y = 0.4*np.sin(2*np.pi*x) + 0.3*np.random.normal(size=x.shape)

x2 = np.linspace(-1.0,1.0,110)
h = 0.4*np.sin(2*np.pi*x2)
plt.scatter(x,y)
plt.plot(x2,h)


X = np.vstack([np.ones(x.shape),x, x**2, x**3, x**4, x**5, x**6, x**7]).T
X2= np.vstack([np.ones(x2.shape),x2, x2**2, x2**3, x2**4, x2**5, x2**6, x2**7]).T

def model_params(X,Y, lam=0.0):
    xx = np.dot(X.T, X)
    xx_inv = np.linalg.inv(xx + lam*np.eye(xx.shape[0]))
    xx_inv_xt = np.dot(xx_inv, X.T)
    params = np.dot(xx_inv_xt,Y)
    return params



plt.scatter(x,y, label = 'Examples')

plt.plot(x2,h, label='Underlying function')

for lam in [0.0,0.00001,0.0001,0.001,0.01]:
    params = model_params(X,y,lam)    
    preds2 = np.dot(X2, params)
    plt.plot(x2,preds2, label='Reg={}'.format(lam))

plt.legend()