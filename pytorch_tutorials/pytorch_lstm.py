# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:47:48 2017

@author: Edward
"""

import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(1)

nn.Conv2d
X = Variable(torch.Tensor(np.random.normal(size=(10,1,24))))

lstm = nn.LSTM(24,4)
hidden = (Variable(torch.randn(1,1,4)),Variable(torch.randn(1,1,4)))
first = nn.Linear(24,24)


out, hidden = lstm(X, hidden)



out = F.relu(first(X))



lstm = nn.LSTM(3, 3)
inputs = Variable(torch.randn((5,1, 3)))
hidden = (Variable(torch.randn(1, 1, 3)), Variable(
    torch.randn((1, 1, 3))))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print('=========================')
print(hidden)
