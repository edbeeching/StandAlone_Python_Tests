# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:29:45 2017

@author: Edward
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
gamma = 0.99

def discount_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    running_acc = 0
    
    for i, reward in reversed(list(enumerate(rewards))):
        running_acc = running_acc*gamma + reward
        discounted_rewards[i] = running_acc       
    return discounted_rewards


class Agent():
    def __init__(self, lr, state_size, action_size, hidden_size):
        
        self.state_in = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
        hidden_layer = slim.fully_connected(self.state_in, 
                                           hidden_size, 
                                           biases_initializer=None,
                                           activation_fn=tf.nn.relu)
        
        self.output = slim.fully_connected(hidden_layer, 
                                           action_size, 
                                           activation_fn=tf.nn.softmax, 
                                           biases_initializer=None)

        self.chosen_action = tf.argmax(self.output, 1)
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indices = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indices)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        tvars = tf.trainable_variables()
        
        self.gradient_holders = []
        
        for idx, var in enumerate(tvars): # is this needed?
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
            
        self.gradients = tf.gradients(self.loss, tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
        
        
# Train the agent
        
tf.reset_default_graph()

agent = Agent(lr=1e-2, state_size=4, action_size=2, hidden_size=8)

total_episodes = 500
max_ep = 999
update_frequency = 5




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    total_reward = []
    total_length = []
    
    grad_buffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(grad_buffer):
        grad_buffer[ix] = grad*0 # remove grad ?
        
        
    for i in range(total_episodes):
        state = env.reset()
        running_reward = 0
        history = []
        
        for j in range(max_ep):
            action_distribution = sess.run(agent.output, 
                                           feed_dict={agent.state_in:[state]})
            action = np.random.choice([0,1],
                                      p=action_distribution[0])
            
            
            next_state, reward,  finished, _ = env.step(action)
            
            history.append([state, action, reward, next_state])
            state = next_state
            running_reward += reward
            
            
            if finished:            
                history = np.array(history)
                history[:, 2] = discount_rewards(history[:, 2])
                feed_dict = {agent.state_in:        np.vstack(history[:, 0]),
                             agent.action_holder:   history[:, 1],
                             agent.reward_holder:   history[:, 2]}
                grads = sess.run(agent.gradients, feed_dict=feed_dict)
                
                for idx, grad in enumerate(grads):
                    grad_buffer[idx] += grad

        
                if i % update_frequency == 0 and i != 0:
                    feed_dict = dict(zip(agent.gradient_holders, grad_buffer))
                    _ = sess.run(agent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(grad_buffer):
                        grad_buffer[ix] = grad*0 # remove grad ?
    
                total_reward.append(running_reward)
                total_length.append(j)
                break


        if i % 100 == 0:
            print(i, np.mean(total_reward[-100:]))


        


        
        
        
        