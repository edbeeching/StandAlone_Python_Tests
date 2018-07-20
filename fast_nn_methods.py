#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:51:40 2018

@author: edward
"""
import numpy as np
import matplotlib.pyplot as plt


from torchvision.datasets import MNIST
from sklearn.neighbors import NearestNeighbors


data = MNIST(root='./', download=True)

all_data = []

for i in range(len(data)):
    all_data.append(np.asarray(data[i][0]))
    
all_data  = np.array(all_data)
all_data = all_data.reshape((60000,-1)) / 255.0


nn_full = NearestNeighbors()
nn_full.fit(all_data)

neighbours = nn_full.kneighbors(all_data[0].reshape(1,-1), 64, return_distance=False)



for i in range(128):

    ax = plt.subplot(8,16, i+1)
    ax.imshow(all_data[neighbours[0,i]].reshape(28,28))
   
    ax.axis('off')


