# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:29:15 2017

@author: Edward
"""


def is_even(n):
    if n < 0: return is_even(-n)
    if n < 2: return 1-n
    
    subs = [1 << i for i, bit in 
            enumerate(bin(n)[-2:1:-1], 1) if 
            is_even(int(bit)) is not True]
    
    return is_even(n - sum(subs))














is_even(2)


for i in range(25):
    print('------------------')
    print(i, is_even(i))
x = reversed(bin(5)[2:])


bin(1)[:1:-1]

is_even(2)
bin(2)[:1:-1]
bin(2)[-2:1:-1]
