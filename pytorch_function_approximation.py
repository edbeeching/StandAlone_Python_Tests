# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 22:00:32 2017

@author: Edward
"""
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
from torch.optim import Adam

class Approximator(nn.Module):
    
    def __init__(self):
        super(Approximator,self).__init__()
        self.hidden = nn.Linear(1, 4)
        self.out = nn.Linear(4, 1)
        
    def forward(self, x):
        x = self.out(F.tanh(self.hidden(x)))
        return x


x = np.linspace(-1.0,1.0,101)
y = np.cos(2*np.pi*0.8*x)+ np.random.normal(size=(101,))

approx = Approximator()
criterion = nn.MSELoss()
optimizer = Adam(approx.parameters())


X_train = Variable(Tensor(x.reshape(-1,1)))
Y_train = Variable(Tensor(y))


for epoch in range(10000):
    
    optimizer.zero_grad()
    out = approx(X_train)

    loss = criterion(out, Y_train)
    loss.backward()
    optimizer.step()
    #print(loss.data[0])


plt.plot(x,y)
plt.plot(x,approx(X_train).data.numpy())




