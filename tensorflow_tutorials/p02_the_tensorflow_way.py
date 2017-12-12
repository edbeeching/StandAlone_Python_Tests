# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:57:48 2017

@author: Edward
"""

# Layer nesting operations
import matplotlib.pyplot as plt 

import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops
ops.reset_default_graph()


sess = tf.Session()


my_array = np.array([[1,3,5,7,9], [-2,0,2,4,6], [-6,3,0,3,6.]])
x_vals = np.array([my_array, my_array])
x_data = tf.placeholder(tf.float32, shape=(3,5))
m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])


prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)
s
for x_val in x_vals:
    print(sess.run(add1, feed_dict={ x_data: x_val}))
    
    
    
#%%  Working with multiple layers
ops.reset_default_graph()

sess = tf.Session()

x_shape = [1,4,4,1]
x_val = np.random.uniform(size=x_shape)

my_filter = tf.constant(0.25, shape=[2,2,1,1])
my_strides = [1,2,2,1]
mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving Average Filter')


def custom_layer(input_matrix):
    input_matrix_squeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1.,2.],[-1,3]])














    
    