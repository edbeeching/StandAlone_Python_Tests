# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:33:15 2017

@author: Edward
"""

import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical


from torch.nn import Conv2d, Module, Linear, CrossEntropyLoss, NLLLoss
from torch.nn.functional import max_pool2d, relu, softmax
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
from more_itertools import chunked
import torch


def preprocess(X, Y):
    X = (X-127.5) / 128.0
    
    Xtorch = X.reshape((-1,1,28,28)).astype(np.float32)
    Ytorch = Y.astype(np.int32)
    return Xtorch, Ytorch

class LeNet(Module):
    
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = Conv2d(1,6,5)
        self.conv2 = Conv2d(6,16,5)
        self.fc1 = Linear(16*4*4, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = max_pool2d(relu(self.conv1(x)), 2)
        x = max_pool2d(relu(self.conv2(x)), 2)
        x = x.view(-1, 16*4*4)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = softmax(self.fc3(x))
        return x
        

def data_generator(X, Y, batch_size=32 ):
    num_examples = X.shape[0]
    
    indices = np.random.permutation(num_examples)
    X = X[indices]
    Y = Y[indices]  
    
    for ids in chunked(range(num_examples),batch_size):
        yield X[ids], Y[ids]
        
        

def train_model(model,  optimizer, criterion,  X,Y, batch_size=32, epochs=1, valid=None):   
    
    for epoch in range(epochs):
        generator = iter(data_generator(X,Y,batch_size))
        num_examples = X.shape[0]
        running_loss = 0.0
        for batch in tqdm(range(0, num_examples, batch_size)):
        
            x, y =  next(generator)
            x, y = Variable(Tensor(x)), Variable(torch.LongTensor(y.tolist()))
            
            optimizer.zero_grad()
            
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if batch%2000 ==0:
                print('Epoch {}, batch {} loss = {}'.format(epoch+1,batch, running_loss/2000.0))
                running_loss = 0.0
        
    print('Finished training')
 
    
leNet = LeNet()
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, Y_train = preprocess(X_train, Y_train)
X_test, Y_test = preprocess(X_test, Y_test)

opti = Adam(leNet.parameters())
crit = CrossEntropyLoss()

train_model(leNet, opti, crit, X_train, Y_train, batch_size=32, epochs=10)


def evaluate(model, X, Y):
    outputs = model(Variable(Tensor(X)))
    _, preds = torch.max(outputs.data, 1)
    preds = preds.numpy()
    
    total = X.shape[0]
    trues = np.sum(preds==Y)
    return trues/total
    

print('LeNet train for 10 epochs')
print('Train accuracy = ', evaluate(leNet, X_train, Y_train))
print('Test accuracy = ', evaluate(leNet, X_test, Y_test))

    







