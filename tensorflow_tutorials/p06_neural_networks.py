# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:23:12 2017

@author: Edward
"""

#%% Implementing gates
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()


sess= tf.Session()

a = tf.Variable(tf.constant(4.))
x_val = 5
x_data = tf.placeholder(dtype=tf.float32)

multiplication = tf.multiply(a, x_data)

loss = tf.square(tf.subtract(multiplication, 50.0))

init = tf.global_variables_initializer()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

print('Running optimization', sess.run(a))

for i in range(10):
    sess.run(train_step, feed_dict={x_data: x_val})
    a_val = sess.run(a)
    mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
    print(a_val, ' * ', x_val, ' = ', mult_output)

sd = tf.constant([1.0,2.0])
r = tf.random_normal(shape=(1,2))
add = tf.add(sd,r)

print(sess.run(add))

#%% activation functions

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()


sess = tf.Session()
#tf.set_random_seed(5)
#np.random_seed(42)

batch_size = 50

a1 = tf.Variable(tf.random_normal(shape=[1,1]))
b1 = tf.Variable(tf.random_normal(shape=[1,1]))
a2 = tf.Variable(tf.random_normal(shape=[1,1]))
b2 = tf.Variable(tf.random_normal(shape=[1,1]))

x = np.random.normal(2,0.1,500)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))
relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

sigmoid_loss = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
relu_loss = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))



sess.run(tf.global_variables_initializer())

opti = tf.train.GradientDescentOptimizer(0.01)

step_sigmoid = opti.minimize(sigmoid_loss)
step_relu = opti.minimize(relu_loss)

print('running optimization')

loss_vec_sigmoid = []
loss_vec_relu = []
for i in range(500):
    rand_indices = np.random.choice(len(x), size=batch_size)
    x_vals = np.transpose([x[rand_indices]])
    sess.run(step_sigmoid, feed_dict={x_data: x_vals})
    sess.run(step_relu, feed_dict={x_data: x_vals})
    
    loss_vec_sigmoid.append(sess.run(sigmoid_loss, feed_dict={x_data: x_vals}))
    loss_vec_relu.append(sess.run(step_relu, feed_dict={x_data: x_vals}))    
    
    sigmoid_output = np.mean(sess.run(sigmoid_activation, feed_dict={x_data: x_vals}))
    relu_output = np.mean(sess.run(relu_activation, feed_dict={x_data: x_vals}))
    
    if i%50==0:
        print('sigmoid = ' + str(np.mean(sigmoid_output)) + ' relu = ' + str(np.mean(relu_output)))
    

# Plot the loss
plt.plot(loss_vec_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(loss_vec_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


print(sess.run(a1))
print(sess.run(a2))
print(sess.run(b1))
print(sess.run(b2))



#%%

import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()


(X,Y) = load_iris(return_X_y=True)
Ytrain_hot = np.array(Y > 0.0, dtype=np.float32)

sess = tf.Session()

x_data = tf.placeholder(shape=[None,4], dtype=tf.float32)
y_data = tf.placeholder(shape=[None,1], dtype=tf.float32)
W1 = tf.Variable(tf.random_normal(shape=[4,5]))
B1 = tf.Variable(tf.random_normal(shape=[1,5]))

W2 = tf.Variable(tf.random_normal(shape=[5,1]))
B2 = tf.Variable(tf.random_normal(shape=[1,1]))

O1 = tf.nn.relu(tf.add(tf.reduce_sum(tf.matmul(x_data, W1)), B1))
O2 = tf.sigmoid(tf.add(tf.reduce_sum(tf.matmul(O1, W2)), B2))

loss = tf.reduce_mean(tf.square(tf.subtract(O2, y_data)))

opti = tf.train.GradientDescentOptimizer(0.001)
update = opti.minimize(loss)
sess.run(tf.global_variables_initializer())

import more_itertools
batch_size = 32
for i in range(1000):
    for chunk in more_itertools.chunked(np.random.permutation(len(Y)), batch_size):
    
        Xbatch = np.array(X[chunk])
        Ybatch = np.expand_dims(Ytrain_hot[chunk],1)
        sess.run(update, feed_dict={x_data: Xbatch, y_data: Ybatch})
    if i % 10 == 0:
       print(i, sess.run(loss, feed_dict={x_data: Xbatch, y_data: Ybatch}))
    
    
sess.run(O2, feed_dict={x_data: X})    














