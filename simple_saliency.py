#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:34:06 2018

@author: edward

Test of generating Grad CAM saliency maps in the simplest possible was with backwards hooks

Applied on VGG19 CNN feature vectors

"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from skimage.io import imread
from torchvision import transforms
from torchvision.utils import make_grid

from cv2 import resize


#print('adding hook')
#
#print(model.features)
#
## load an image
#
#img = imread('both.png')
#plt.imshow(img)
#img = preproc(img)

#assert 0 
#
#import json
#class_idx = json.load(open("imagenet_class_index.json"))
#idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
#
#
#grid = make_grid(forward_outputs[0].data.view(512,1,14,14), nrow=32).numpy().transpose(1,2,0)
#grid_back = make_grid(backward_outputs[0].data.view(512,1,14,14), nrow=32).numpy().transpose(1,2,0)
#
#plt.imshow(grid[:,:,0])
#plt.imshow(grid_back[:,:,0])

#plt.imshow(F.relu(Variable(res)).data.numpy())
#
#upscaled = resize(F.relu(Variable(res)).data.numpy(), (224, 224))
#
#plt.imshow(imread('both.png'))
#
#plt.imshow(upscaled, alpha=0.9, cmap='gray')

def get_salency(model, image, layer=-2, index=None):
    preproc = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img = preproc(image)
    forward_outputs  = []
    backward_outputs  = []    
    
    def forward_hook(self, input, output):
        forward_outputs.append(output[0].clone())
        
        
    def backward_hook(self, grad_input, grad_output):
        backward_outputs.append(grad_output[0].clone())    
        
    model.features[layer].register_forward_hook(forward_hook)
    model.features[layer].register_backward_hook(backward_hook) 
      
    preds = model(Variable(img.unsqueeze(0)))
    
    if index == None:    
        index = np.argmax(preds.data.numpy())
    gradient_vector = torch.zeros_like(preds)
    gradient_vector[0, index] = 1.0
    
    loss = (preds*gradient_vector).sum()
    loss.backward()    

    forwards = forward_outputs[0].data
    backwards = backward_outputs[0][0].data
 
    res = (forwards * backwards).mean(dim=0) 

    upscaled = resize(F.relu(Variable(res)).data.numpy(), (224, 224))
    
    return upscaled

model = torchvision.models.vgg19(pretrained=True).eval()
image = resize(imread('cat.jpg'), (224,224))

saliency = get_salency(model, image)

plt.imshow(image)
plt.imshow(saliency, alpha=0.9, cmap='gray')
