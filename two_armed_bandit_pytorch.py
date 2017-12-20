# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:36:56 2017

@author: Edward
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.autograd import Variable


def pull_bandit(bandit):    
    if np.random.randn(1) >  bandit:
        return 1 # Positive reward
    else: 
        return -1 # Negative reward
    
bandits = [0.2, 0.0, -0.2, -5.0]
num_bandits = len(bandits)
total_episodes = 1000
total_reward = np.zeros(num_bandits)
e = 0.1
total_reward = np.zeros([1,4])
#policy = Policy(num_bandits)
## Define policy 
weights = Variable(torch.ones(1,num_bandits), requires_grad=True)
optimizer = torch.optim.SGD([weights], lr=0.001) 
for i in range(1000):
    if np.random.rand(1) < e:
        action = np.random.choice(num_bandits)
    else:
        _, action = torch.max(weights, 0)
        action = int(action.data.numpy()[0])
        #action = int(policy.best_action().data.numpy())
    
    reward = pull_bandit(bandits[action])
    total_reward[0,action] +=  reward
    #policy.update(Tensor(action), Tensor(reward))
    optimizer.zero_grad()
    loss = -(torch.log(weights[0,action])*Variable(Tensor([reward])))
    loss.backward()
    optimizer.step()   
    if i%50 == 0:
        print(total_reward)
    
    
print(weights)
    
    
    