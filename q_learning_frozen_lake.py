# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:49:44 2017

@author: Edward
"""
import numpy as np
import matplotlib.pyplot as plt

import gym

env = gym.make('FrozenLake-v0')

Qtable = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = 0.8
gamma = 0.95
num_episodes = 2000

reward_list = []

for i in range(num_episodes):
    state = env.reset()
    all_rewards = 0
    for j in range(100):
        action = np.argmax(Qtable[state,:]+np.random.randn(1, env.action_space.n)*(1.0/(i+1)))
        next_state, reward, done,_ = env.step(action)
        Qtable[state, action] += learning_rate*(reward + gamma*np.max(Qtable[next_state,:]) - Qtable[state, action])
        all_rewards += reward
        state = next_state
        
        if done:
            break
    reward_list.append(all_rewards)