# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:26:05 2017

@author: Edward
"""

from keras.applications import vgg19
from scipy.misc import imread
import scipy
import numpy as np
import matplotlib.pyplot as plt
model = vgg19.VGG19()

image = imread('C:/Users/Edward/Documents/PycharmProjects/FlaskApp/FlaskApp/images/zebra.jpg')
image = scipy.misc.imresize(image, (224, 224))
image = np.expand_dims(image, 0).astype('float32')
image = vgg19.preprocess_input(image)

preds = model.predict(image)
decodes = vgg19.decode_predictions(preds)
for p in decodes:
    print(p, '--------')


def make_prediction(filepath):
    def crop_with_ratio(img, size=(224, 224)):
        # function that takes in input image and crops and resizes to maintain aspect ratio
        height, width, chans = image.shape
        if height == width:
            return scipy.misc.imresize(image, size)
        ratio = height / width
        if ratio < 1.0:  # width is greater than height
            diff = width - height
            border_size = diff // 2
            return scipy.misc.imresize(image[:, border_size:-border_size, :], size)
        else:
            diff = height - width
            border_size = diff // 2
            return scipy.misc.imresize(image[border_size:-border_size, :, :], size)


    image = imread(filepath)
    image = crop_with_ratio(image)
    image = np.expand_dims(image, 0).astype('float32')
    image = vgg19.preprocess_input(image)

    preds = model.predict(image)
    decodes = vgg19.decode_predictions(preds)
    for p in decodes:
        print(p, '--------')
        
make_prediction('C:/Users/Edward/Documents/PycharmProjects/FlaskApp/FlaskApp/images/zebra.jpg')
        
img = crop_with_ratio(image)      
        
plt.figure()
plt.subplot(1,3,1)
plt.imshow(image)       
plt.subplot(1,3,2)
plt.imshow(scipy.misc.imresize(image, (224, 224)))               
plt.subplot(1,3,3)
plt.imshow(img)        
    
    
    
    
    
    