# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:26:05 2017

@author: Edward
"""

from keras.applications import vgg19
from scipy.misc import imread
import scipy
import numpy as np
model = vgg19.VGG19()

image = imread('C:/Users/Edward/Documents/PycharmProjects/FlaskApp/FlaskApp/images/zebra.jpg')
image = scipy.misc.imresize(image, (224, 224))
image = np.expand_dims(image, 0).astype('float32')
image = vgg19.preprocess_input(image)

preds = model.predict(image)
decodes = vgg19.decode_predictions(preds)
for p in decodes:
    print(p, '--------')


def crop_with_Ratio(image, size=(224,224)):
    