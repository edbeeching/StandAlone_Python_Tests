# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:33:16 2017

@author: Edward
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = (X_train / 128.0) - 1.0
X_train = X_train.astype(np.float32)
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected, max_pool2d, flatten

# Define the network
model_input = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)

conv1 = conv2d(model_input, 32, (3,3), stride=1, padding='SAME')
pool1 = max_pool2d(conv1, (2,2))

conv2 = conv2d(pool1, 32, (3,3), stride=1, padding='SAME')
pool2 = max_pool2d(conv2, (2,2))
fc1_in = flatten(pool2)
fc1 = fully_connected(fc1_in, 256)
fc2 = fully_connected(fc1, 256)
fc3 = fully_connected(fc2, 256)
model_output = fully_connected(fc3, 10, activation_fn=None)
target = tf.placeholder(dtype=tf.int32, shape=(None, 1))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=tf.one_hot(target,10)))

optimizer = tf.train.AdamOptimizer()
update = optimizer.minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    batch_size = 64
    num_examples = len(Y_train)
    num_splits =  num_examples // batch_size   
    for epoch in range(10):
        permute = np.random.permutation(len(Y_train))
        Xperm = X_train[permute]
        Yperm = Y_train[permute]
        
        losses = []
#        i=0
        for batch_x, batch_y in zip(np.array_split(Xperm, num_splits),
                                    np.array_split(Yperm, num_splits)):
#            if i%50==0:
#                print(i)
#            i+=1
            feed_dict = {model_input: batch_x.reshape(-1,28,28,1),
                         target: batch_y.astype(np.int32).reshape(-1,1)}
            _, L = sess.run([update, loss], feed_dict=feed_dict)        
            losses.append(L.reshape(1,-1))
            
        
        print(np.mean(np.hstack(losses)))
    save_path = saver.save(sess, "./model.ckpt")

    
    
with tf.Session() as sess:
    saver.restore(sess, "./model.ckpt")
    feed_dict = {model_input: X_train[0:100].reshape(-1,28,28,1)}    
    preds = sess.run(model_output, feed_dict)


maxs = np.argmax(preds,1)

tt = np.vstack([maxs, Y_train[0:100]])

with tf.Session() as sess:
    hots = sess.run(tf.one_hot(batch_y,10))
