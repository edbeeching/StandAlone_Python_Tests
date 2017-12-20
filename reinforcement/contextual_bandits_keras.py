# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 23:06:35 2017

@author: Edward
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:18:07 2017

@author: Edward
"""

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

class ContentualBandit():
    def __init__(self):
        self.state = 0;
        self.bandits = np.array([[0.2,0.0,0.0,-2],[0.1,-5,1.0,0.25],[-5.0,5.0,5.0,5.0]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def get_bandit(self):
        self.state = np.random.randint(0, self.num_bandits)
        return self.state
    
    def pull_arm(self, action):
        bandit = self.bandits[self.state, action]
        if np.random.randn(1) > bandit:
            return 1
        else:
            return -1
        
        
from keras.models import Model
from keras.layers import Input, Dense
import keras.backend as K


class Agent():
    def __init__(self, lr, s_size, a_size):
        
        model_input = Input(shape=(s_size,))
        action_layer = Dense(a_size, use_bias=False, kernel_initializer='ones')(model_input)
        action = K.argmax(action_layer)
        responsible_weight = K.sl
        
        
        
        
        self.state_in =  tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_onehot = slim.one_hot_encoding(self.state_in, s_size)
        output = slim.fully_connected(state_in_onehot, 
                                      a_size, 
                                      biases_initializer=None,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.arg_max(self.output, 0 )
        
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)
        

# training the agent

tf.reset_default_graph()

contentual_bandit = ContentualBandit()
agent = Agent(lr=0.001, 
              s_size=contentual_bandit.num_bandits,
              a_size=contentual_bandit.num_actions)
weights = tf.trainable_variables()[0]
total_episodes = 10000
total_reward = np.zeros([contentual_bandit.num_bandits,
                          contentual_bandit.num_actions])

e = 0.1


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    for i in range(total_episodes):
        state = contentual_bandit.get_bandit()
        
        if np.random.rand(1) < e:
            action = np.random.randint(contentual_bandit.num_actions)
        else:
            action = sess.run(agent.chosen_action,
                              feed_dict={agent.state_in: [state]} ) 
        reward = contentual_bandit.pull_arm(action)
        
        # Update the network based on the results
        
        feed_dict = {agent.reward_holder:[reward],
                     agent.action_holder:[action],
                     agent.state_in: [state]}
        _, ww = sess.run([agent.update, weights], 
                         feed_dict=feed_dict)
        
        total_reward[state, action] += reward
        if i% 500 == 0:
            print('Mear reward', str(np.mean(total_reward, axis=1)))



for a in range(contentual_bandit.num_bandits):
    print("The agent thinks action " + str(np.argmax(ww[a])+1) + " for bandit " + str(a+1) + " is the most promising....")
    if np.argmax(ww[a]) == np.argmin(contentual_bandit.bandits[a]):
        print("...and it was right!")
    else:
        print("...and it was wrong!")










        
        
        
        
        
        
        
        