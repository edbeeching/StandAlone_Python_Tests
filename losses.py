#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:38:48 2018

@author: edward
"""


import math
import numpy as np
import matplotlib.pyplot as plt
def binary_cross_entropy(y, yhat):
    
    return  -y*math.log(yhat) - (1.0-y)*math.log(1-yhat)



def categorical_cross_entropy(y, y_hat):
    # y = [0.0,1.0] yhat = [0.1,0.9] for example
    
    loss= 0.0
    
    for p,q in zip(y,yhat):
        loss -= p*math.log(q)
        
    return loss
        
    
    



if __name__ == '__main__':
    
    yhats = np.linspace(0.1, 0.9, 10)
    losses = []
    
    
    for yhat in yhats:
        loss = binary_cross_entropy(1.0, yhat)
        print(yhat, loss)
        losses.append(loss) 
        
    
    plt.plot(yhats, losses)
    
    
    y = [0.0, 1.0]
    y_hats = [[0.1,0.9], [0.2,0.8], [0.3,0.7], [0.4,0.6], [0.5,0.5],
              [0.6,0.4], [0.7,0.3], [0.8,0.2], [0.9,0.1]]
    
    for yhat in y_hats:
        print(yhat, categorical_cross_entropy(y, yhat))
    
    
    

