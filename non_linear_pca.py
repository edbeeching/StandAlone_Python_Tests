# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:04:07 2017

@author: Edward
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Model
from keras.layers import Dense, Input
from keras.regularizers import l2

def gen_data(num=1000):

    x = np.linspace(-np.pi,np.pi,1001)
    
    x1 = np.sin(x)+np.random.normal(scale=0.1, size=x.shape)
    x2 = np.cos(x)+np.random.normal(scale=0.1, size=x.shape)
    return np.vstack([x1, x2]).T

X = gen_data()




model_input = Input(shape=(2,))
x = Dense(8, activation='tanh', kernel_regularizer=l2(0.0001))(model_input)
x = Dense(4, activation='tanh', kernel_regularizer=l2(0.0001))(x)
encoded = Dense(1, activation='linear')(x)
x = Dense(4, activation='tanh', kernel_regularizer=l2(0.0001))(encoded)
x = Dense(8, activation='tanh', kernel_regularizer=l2(0.0001))(x)
decoded = Dense(2, activation='linear')(x)


autoencoder = Model(model_input, decoded)
encoder = Model(model_input, encoded)

autoencoder.compile(optimizer='adam',loss='mse')


autoencoder.fit(X,X, epochs=5000)

Xp = autoencoder.predict(X)
zp = encoder.predict(X)


plt.scatter(X[:,0], X[:,1])
plt.scatter(Xp[:,0], Xp[:,1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], zp)


