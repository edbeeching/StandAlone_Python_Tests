# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:04:16 2017

@author: Edward
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.merge import _Merge
import keras.backend as K
from functools import partial
# Data generation
BATCH_SIZE = 1
GRADIENT_PENALTY_WEIGHT = 10.0


def eight_gaussians(num, scale):
    centres = [[0.0, 0.7],
               [0.0, -0.7],
               [-0.7, 0.0],
               [0.7, 0.0],
               [0.5, 0.5],
               [-0.5, 0.5],
               [-0.5, -0.5],
               [0.5, -0.5]]
    
    positions = []           
    
    for centre in centres:
        sample = np.random.normal(loc = centre, scale=scale, size=(num, 2))
        positions.append(sample)
    
    return np.array(positions).reshape((-1,2))



def wasserstein_loss(y_true, y_pred):
    
    return K.mean(y_true * y_pred)
    
def gradient_penalty_loss(y_true, y_pred, average_samples, weight):
    gradients = K.gradients(K.sum(y_pred), average_samples)
    gradients_norm = K.sqrt(K.sum(K.square(gradients)))
    penalty = weight * K.square(1 - gradients_norm)
    return penalty
    
    



def make_generator():
    # Create the generator
    gen_input = Input(shape=(100,))
    x = Dense(40, activation='relu')(gen_input)
    x = Dense(40, activation='relu')(x)
    x = Dense(40, activation='relu')(x)
    gen_output = Dense(2, activation='linear')(x)
    
    generator = Model(gen_input, gen_output)
    generator.summary()
    
    return generator


def make_critic():
    d_input = Input(shape=(2,))
    x = Dense(30, activation='relu')(d_input)
    x = Dense(30, activation='relu')(x)
    x = Dense(30, activation='relu')(x)
    d_output = Dense(1, activation='linear')(x)
    
    critic = Model(d_input, d_output)
    critic.summary()
    
    return critic
    
def make_gan(generator, critic):
    for layer in critic.layers:
        layer.trainable = False
    
    gen_input = Input(shape=(100,))
    gen_out = generator(gen_input)
    critic_out = critic(gen_out)
    
    critic.trainable = False

    gan = Model(gen_input, critic_out)

    gan.compile(optimizer='adam', loss = wasserstein_loss)
    gan.summary()
      
    critic.trainable = True
    for layer in critic.layers:
        layer.trainable = True
    
    
    for layer in generator.layers:
        layer.trainable = False
    generator.trainable = False
    
    return gan

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1,1,1))
        return (weights*inputs[0]) + ((1-weights)*inputs[1])


if __name__ == '__main__':
    
    generator = make_generator()
    critic = make_critic()
    generator.compile(optimizer='adam', loss=wasserstein_loss)
    gan = make_gan(generator, critic)
    
    # Get some real data
    real_data = eight_gaussians(1000, 0.1)
    
    # Set up the discriminator model
    
    real_samples = Input(shape=(2,))
    
    input_for_gen = Input(shape=(100,))
    input_for_critic = generator(input_for_gen)
    
    critic_output_for_gen = critic(input_for_critic)
    critic_output_for_real = critic(real_samples)
    
    average_samples = RandomWeightedAverage()([real_samples, input_for_critic])
    average_samples_out = critic(average_samples)

    # use a partial function to get around the loss function issue
    
    partial_gp_loss = partial(gradient_penalty_loss,
                              average_samples=average_samples,
                              weight=GRADIENT_PENALTY_WEIGHT)

    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model(inputs = [real_samples, input_for_gen], 
                         outputs=[critic_output_for_real, critic_output_for_gen, average_samples_out])
    
    critic_model.compile(optimizer='adam', loss = [wasserstein_loss, wasserstein_loss, partial_gp_loss])
    critic_model.summary()
    
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y =  np.ones((BATCH_SIZE, 1), dtype=np.float32)*-1.0
    dummy_y = np.zeros((BATCH_SIZE,1), dtype=np.float32)
    
    
    for epoch in range(100):
        
        np.random.shuffle(real_data)
        num_examples = real_data.shape[0]
        critic_steps = 5
        minibatch_size = BATCH_SIZE * critic_steps
        examples_per_minibatch = int(num_examples / minibatch_size)
        print('Epoch', epoch)
        
#        discriminator_loss = []
#        generator_loss = []
        
        for i in range(examples_per_minibatch):
            minibatch = real_data[i*minibatch_size: (i+1)*minibatch_size, :]
            for j in range(critic_steps):
                data_batch = minibatch[j*BATCH_SIZE: (j+1)*BATCH_SIZE]
                noise = np.random.uniform(size=(BATCH_SIZE, 100)).astype(np.float32)
                critic_model.train_on_batch([data_batch, noise], [positive_y, negative_y, dummy_y])
            gan.train_on_batch(np.random.uniform(size=(BATCH_SIZE, 100)), positive_y)
            




    xx = generator.predict(np.random.uniform(size=(100, 100)))







