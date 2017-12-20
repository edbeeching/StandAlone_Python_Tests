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
import torch

class Approximator(nn.Module):
    
    def __init__(self):
        super(Approximator,self).__init__()
        self.hidden = nn.Linear(1, 32)
        self.out = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.out(F.tanh(self.hidden(x)))
        return x

def l2_loss(reg, model):   
    square_2_norm = 0
    
    for name, p in model.named_parameters():
         if 'weight' in name:
             square_2_norm += torch.norm(p, 2)**2
    return reg*square_2_norm
        
x = np.linspace(-1.0,1.0,101)
y = np.cos(2*np.pi*0.8*x)+ 0.1*np.random.normal(size=(101,))

approx = Approximator()
criterion = nn.MSELoss()
optimizer = Adam(approx.parameters())


X_train = Variable(Tensor(x.reshape(-1,1)))
Y_train = Variable(Tensor(y))


for epoch in range(10000):
    optimizer.zero_grad()
    out = approx(X_train)

    loss = criterion(out, Y_train) + l2_loss(0.0001, approx)
    loss.backward()
    optimizer.step()


plt.plot(x,y)
plt.plot(x,approx(X_train).data.numpy())


out = approx(X_train)
out99 = Variable(Tensor(out.data.numpy()*0.99))
loss = criterion(out, out99)
loss.backward()
for name, p in approx.named_parameters():
     print(p.grad)
