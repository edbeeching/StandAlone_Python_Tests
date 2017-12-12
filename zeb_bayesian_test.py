# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:35:16 2017

@author: Edward
"""

import numpy as np
t = np.eye(5)

for sub in np.array_split(t,3):
    print(sub)
    print('##################')
