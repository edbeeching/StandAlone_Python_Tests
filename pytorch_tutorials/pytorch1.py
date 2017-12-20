# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:24:48 2017

@author: Edward
"""

import torch




x = torch.rand(5,3)
y = torch.rand(5,3)

print(x+y)

print(x[:,1])

a = torch.ones(5)
print(a)


b=a.numpy()


# Autograd

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2,2), requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 3
out = z.mean()

out.backward()
print(x.grad)




