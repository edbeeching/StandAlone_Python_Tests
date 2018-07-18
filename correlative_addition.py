#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:29:04 2018

@author: edward
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable


def correlative_addition(base, additive):
    
    assert base.size() == additive.size()
    
    xcor = base * additive
    xcor_relu = F.relu(xcor)
    mask = xcor_relu > 0.0
    
    return base + additive*mask.float()


output1 = Variable(torch.randn(4,10), requires_grad=True)
output2 = Variable(torch.randn(4,10), requires_grad=True)

added = correlative_addition(output1, output2)

print(output1)
print(output2)
print(added)

target = torch.zeros_like(added)

loss = (added - target).sum()
loss.backward()

print(output1.grad)
print(output2.grad)




    

    
    