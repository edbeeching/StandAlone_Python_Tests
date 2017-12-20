# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:05:24 2017

@author: Edward
"""


import numpy as np
import pickle
import tensorflow as tf
import math

import gym

tf.reset_default_graph()
env = gym.make('CartPole-v0')

# =============================================================================
#                               Hyperparameters
# =============================================================================

policy_hidden_units = 8
model_hidden_units  = 256
learning_rate       = 0.01
gamma               = 0.99
decay_rate          = 0.99
model_batch_size    = 3
real_batch_size     = 3
state_size          = 4


initializer = tf.contrib.layers.xavier_initializer()
# =============================================================================
#                               Helper Functions
# =============================================================================
def reset_gradient_buffer(grad_buffer):
    for i, grad in enumerate(grad_buffer):
        grad_buffer[i] = grad*0
    return grad_buffer

def discount_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    running_acc = 0
    
    for i, reward in reversed(list(enumerate(rewards))):
        running_acc = running_acc*gamma + reward
        discounted_rewards[i] = running_acc       
    return discounted_rewards


def step_model(sess, graph_input, graph, state_history, action):
    feed_dict = {graph_input: np.reshape(np.hstack([state_history[-1][0], 
                                                       np.array(action)]), [1,5])}
    prediction = sess.run([graph], feed_dict=feed_dict)
    reward = prediction[0][:,4]
    obs = prediction[0][:,0:4]
    obs[:,0] = np.clip(obs[:,0], -2.4, 2.4)
    obs[:,2] = np.clip(obs[:,2], -0.4, 0.4)
    done = np.clip(prediction[0][:,5],0,1)
    if done > 0.1 or len(state_history) >= 300:
        done = True
    else:
        done = False
    return obs, reward, done
    
    
# =============================================================================
#                               Policy network
# =============================================================================

observations = tf.placeholder(tf.float32, [None, 4])

W1 = tf.Variable(initializer([4, policy_hidden_units]))
W2 = tf.Variable(initializer([policy_hidden_units, 1]))

layer1 = tf.nn.relu(tf.matmul(observations, W1))
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()

input_action = tf.placeholder(tf.float32, [None, 1])
advantages = tf.placeholder(tf.float32)
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32)
W2Grad = tf.placeholder(tf.float32)
batch_grad = [W1Grad, W2Grad]
log_likelyhood = tf.log(input_action*(input_action - probability) + 
                        (1 - input_action)*(input_action + probability))

loss = -tf.reduce_mean(log_likelyhood*advantages)
new_grads = tf.gradients(loss, tvars)
update_grads = adam.apply_gradients(zip(batch_grad, tvars))

# =============================================================================
#                               Model Network
# =============================================================================

previous_state_action = tf.placeholder(tf.float32, [None, 5])

W1M = tf.Variable(initializer([5, model_hidden_units]))
W2M = tf.Variable(initializer([model_hidden_units, model_hidden_units]))

B1M = tf.Variable(tf.zeros([model_hidden_units]))
B2M = tf.Variable(tf.zeros([model_hidden_units]))

W_obs = tf.Variable(initializer([model_hidden_units, 4]))
W_reward = tf.Variable(initializer([model_hidden_units, 4]))
W_done = tf.Variable(initializer([model_hidden_units, 4]))

B_obs = tf.Variable(tf.zeros([4]))
B_reward = tf.Variable(tf.zeros([1]))
B_done = tf.Variable(tf.zeros([1]))


layer1M = tf.nn.relu(tf.matmul(previous_state_action, W1M) + B1M)
layer2M = tf.nn.relu(tf.matmul(layer1M, W2M) + B2M)

predicted_obs = tf.matmul(layer2M, W_obs) + B_obs
predicted_reward = tf.matmul(layer2M, W_reward) + B_reward
predicted_done = tf.matmul(layer2M, W_done) + B_done

true_obs = tf.placeholder(tf.float32,[None,4])
true_rewards = tf.placeholder(tf.float32,[None,1])
true_dones = tf.placeholder(tf.float32,[None,1])

predicted_state = tf.concat([predicted_obs, 
                             predicted_reward, 
                             predicted_done],1)

obs_loss = tf.square(true_obs - predicted_obs)
reward_loss = tf.square(true_rewards - predicted_reward)
done_loss = -tf.log(predicted_done*true_dones + (1-predicted_done)*(1-true_dones))

model_loss = tf.reduce_mean(obs_loss + done_loss + reward_loss)
model_adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
update_model = model_adam.minimize(model_loss)


# =============================================================================
#                   Training the Policy and Model
# =============================================================================

obs_history = []
rewards = []
actions = []
dones = []
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
draw_from_model = False
train_model= True
train_policy = False
switch_point = 1
num_episodes = 100
batch_size = real_batch_size


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gradient_buffer = sess.run(tvars)
    gradient_buffer = reset_gradient_buffer(gradient_buffer)
    observation = env.reset()

    while episode_number <= 5000:
        observation = np.reshape(observation, [1,4])
        tfprob = sess.run(probability, feed_dict={observations: observation})
        action = 1 if np.random.uniform() < tfprob else 0
        
        obs_history.append(observation)
        y = 1 if action == 0 else 0
        actions.append(y)

        if not draw_from_model:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done, info = step_model(action)
        
        reward_sum += reward
        dones.append(done*1)
        rewards.append(reward)
        
        
        if done:
            #print('done')
            if not draw_from_model:
                real_episodes += 1
            episode_number += 1
            episode_obs = np.vstack(obs_history)
            episode_actions = np.vstack(actions)
            episode_rewards = np.vstack(rewards)
            episode_dones = np.vstack(dones)
            
            obs_history = []
            rewards = []
            actions = []
            dones = []       
        
            if train_model:
                
                tr_actions = np.array([np.abs(y-1) for i in episode_rewards][:-1])
                previous_states = np.hstack([episode_obs[:-1, :], tr_actions.reshape(-1,1)])
                next_states = episode_obs[1:, :]
                tr_rewards = np.array(episode_rewards[1:, :])
                tr_dones = np.array(episode_dones[1:, :])
                combined = np.hstack([next_states, tr_rewards, tr_dones])
                
                feed_dict = {previous_state_action: previous_states,
                             true_obs: next_states,
                             true_dones: tr_dones,
                             true_rewards: tr_rewards}
                
                lost, pred_state, _ = sess.run([model_loss, predicted_state, update_model],
                                               feed_dict=feed_dict)
            if train_policy:
                discounted_rewards = discount_rewards(episode_rewards).astype('float32')
                discounted_rewards -= np.mean(discounted_rewards)
                discounted_rewards /= np.std(discounted_rewards)
                
                grads = sess.run(new_grads, feed_dict={observations: episode_obs,
                                                       input_action: episode_actions,
                                                       advantages: discounted_rewards})
                if np.sum(grads[0] == grads[0]) == 0:
                    break
                for i, grad in enumerate(grads):
                    gradient_buffer[i] += grad
                
            if (switch_point + batch_size) == episode_number:
                switch_point = episode_number
                if train_policy:
                    sess.run(update_grads, feed_dict={W1Grad: gradient_buffer[0], 
                                                      W2Grad: gradient_buffer[1]})
                    gradient_buffer = reset_gradient_buffer(gradient_buffer)
        
        
                if running_reward is None:
                    running_reward = reward_sum
                else:
                    running_reward = running_reward*0.99 + reward_sum*0.01
                
                
                print('World perf: Ep {}, Reward {:4.2f}, Action {}, Mean Reward {:4.2f}'.format(real_episodes,
                      reward_sum / real_batch_size, action, running_reward/real_batch_size))
                reward_sum = 0
                if episode_number > 100:
                    train_model = not train_model
                    train_policy = not train_policy
                
            if draw_from_model:
                observation = np.random.uniform(-0.1,0.1,[4])
                batch_size = model_batch_size
            else:
                observation = env.reset()
                batch_size = real_batch_size


import matplotlib.pyplot as plt
plt.figure(figsize=(8, 12))
for i in range(6):
    plt.subplot(6, 2, 2*i + 1)
    plt.plot(pred_state[:,i])
    plt.subplot(6,2,2*i+1)
    plt.plot(combined[:,i])
plt.tight_layout()