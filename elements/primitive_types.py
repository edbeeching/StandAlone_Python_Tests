#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:55:07 2018

@author: edward
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

def count_bits(x):
    num_bits = 0
    
    while x:
        num_bits += x & 1
        x >>= 1
    return num_bits

def test_count_bits():
    for i in range(256):
        print(i, bin(i), count_bits(i))
        
        
def clear_nth_bit(x, n):
    
    mask = ~(1<< n)
    return x&mask

def test_clear_nth_bit():
    for j in range(0,2):
        for i in range(256):
            print(j, i, bin(i), bin(clear_nth_bit(i, j)))
            
            
test_clear_nth_bit()

    
    