# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 21:03:57 2017

@author: Edward
"""


from keras.applications import vgg16
from scipy.ndimage import imread
from scipy.misc import imresize
import numpy as np
model = vgg16.VGG16(include_top=False)

plt.figure()
plt.imshow(image)
image = imread('img_532.jpg')

image_proc = vgg16.preprocess_input(image.astype(dtype=np.float32))


features = model.predict(np.expand_dims(image_proc,0))

import matplotlib.pyplot as plt

plt.figure()
for i in range(12):
    plt.subplot(3,4,i+1)
    feature = features[0,:,:,i]
    plt.imshow(np.reshape(feature,(9,12)))

mask = np.zeros((9,12))
    
resized = imresize(image, (9,12))

plt.imshow(resized)

Y = np.reshape(mask, (9*12, 1))
X = np.reshape(features, (9*12, 512))

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X,Y)

preds = clf.predict(X)







