# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:18:46 2017

@author: Edward
"""
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Dense

X = np.linspace(-10*np.pi, 10*np.pi, 4000)
Y = np.sin(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y,random_state=9, test_size=0.5)

plt.figure()
num_neurons = [8,16,32,64,128,256]
for i, neurons in enumerate(num_neurons):
    plt.subplot(2,3,i+1)
    plt.title(str(neurons))
    model_input = Input(shape=(1,))
    x = Dense(neurons, activation='tanh')(model_input)
    x = Dense(neurons, activation='tanh')(x)
    model_output = Dense(1)(x)
    
    model = Model(model_input, model_output)
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.summary()

    model.fit(Xtrain, Ytrain, epochs=500)

    Ypreds = model.predict(X) 
    plt.plot(X,Y)
    plt.scatter(X, Ypreds)
plt.show()



#%%



