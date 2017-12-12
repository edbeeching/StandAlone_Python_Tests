# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:53:07 2017

@author: Edward
"""

import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()

my_tensor = tf.zeros([1,20])

sess.run(my_tensor) # Evaluates my_tensor

my_var = tf.Variable(tf.zeros([1,20]))
sess.run(my_var.initializer) # intialises the variable
sess.run(my_var)


row_dim = 2
col_dim = 3

zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))
one_var = tf.Variable(tf.ones([row_dim, col_dim]))
sess.run(zero_var.initializer)
sess.run(one_var.initializer)

print(sess.run(zero_var))
print(sess.run(one_var))

zero_similar = tf.Variable(tf.zeros_like(zero_var))
one_similar = tf.Variable(tf.ones_like(one_var))

sess.run(zero_similar.initializer)
sess.run(one_similar.initializer)

print(sess.run(zero_similar))
print(sess.run(one_similar))

fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))
sess.run(fill_var.initializer)
print(sess.run(fill_var))

const_var = tf.Variable(tf.constant([8,7,6,5,4,3,2,1]))
const_fil_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))

sess.run(const_fil_var.initializer)
sess.run(const_var.initializer)

print(sess.run(const_fil_var))

# Sequences and ranges
linear_var = tf.Variable(tf.linspace(start=0.0,stop=1.0,num=3))
sequence_var = tf.Variable(tf.range(start=6,limit=15, delta=3))
sess.run(linear_var.initializer)
sess.run(sequence_var.initializer)

print(sess.run(linear_var))
print(sess.run(sequence_var))

# Random number tensors

rnorm_var = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
runif_var = tf.random_uniform([row_dim, col_dim], minval=0.0, maxval=4.0)
print(sess.run(rnorm_var))
print(sess.run(runif_var))

# Visualize the variable creation
ops.reset_default_graph()

sess = tf.Session()

my_var = tf.Variable(tf.zeros([1,20]))

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('/tmp/variable_logs', graph=sess.graph)
initialize_op = tf.global_variables_initializer()

sess.run(initialize_op)











