# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:38:36 2017

@author: Edward

Example 2 armed bandit RL problem

"""


import numpy as np


bandits = [0.2, 0.0, -0.2, -5.0]
num_bandits = len(bandits)

def pull_bandit(bandit):
    
    if np.random.randn(1) >  bandit:
        return 1 # Positive reward
    else: 
        return -1 # Negative reward
    
results = {}
    
for bandit in bandits:
    results[bandit] = []
    for i in range(100):
        results[bandit].append(pull_bandit(bandit))
        
for bandit in bandits:
    print(bandit, np.mean(results[bandit]), np.std(results[bandit]))


# Define the agent
import tensorflow as tf
tf.reset_default_graph()
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights)

reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder,[1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

total_episodes = 1000
total_reward = np.zeros(num_bandits)
e = 0.5

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(total_episodes):
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action) # get an action which is the amx of weights

        reward = pull_bandit(bandits[action])
        _, resp, ww = sess.run([update, responsible_weight, weights], 
                               feed_dict={reward_holder:[reward],
                                          action_holder:[action]})
        total_reward[action] += reward
        if i % 50 == 0:
            print('==========================')
            print(str(total_reward))
            print('The agent thinks: {}, the right answer is {}'.format(
                    str(np.argmax(ww)+1), str(np.argmax(-np.array(bandits))+1)))
    
    
    print(sess.run(weights))

    






