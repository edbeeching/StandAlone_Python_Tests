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

forward_inputs  = []
forward_outputs  = []
backward_inputs  = []
backward_outputs  = []


def forward_hook(self, input, output):
    print(input[0].size())
    print(output[0].size())
    
    forward_inputs.append(input[0].clone())
    forward_outputs.append(output[0].clone())
    
    
def backward_hook(self, grad_input, grad_output):
    print(grad_input[0].size())
    print(grad_output[0].size())
    backward_inputs.append(grad_input[0].clone())
    backward_outputs.append(grad_output[0].clone())

preproc = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

model = torchvision.models.vgg19(pretrained=True).eval()
print('adding hook')
model.features[-2].register_forward_hook(forward_hook)
model.features[-2].register_backward_hook(backward_hook)

print(model.features)

# load an image

img = imread('horse.jpg')
plt.imshow(img)
img = preproc(img)

preds = model(Variable(img.unsqueeze(0)))

index = np.argmax(preds.data.numpy())
gradient_vector = torch.zeros_like(preds)
gradient_vector[0, index] = 1.0

loss = (preds*gradient_vector).sum()
loss.backward()



assert 0 

import json
class_idx = json.load("imagenet_class_index.json")
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


grid = make_grid(forward_outputs[0].data.view(512,1,14,14), nrow=32).numpy().transpose(1,2,0)
grid_back = make_grid(backward_outputs[0].data.view(512,1,14,14), nrow=32).numpy().transpose(1,2,0)

plt.imshow(grid[:,:,0])
plt.imshow(grid_back[:,:,0])

forwards = forward_outputs[0].data
backwards = backward_outputs[0][0].data

res = (forwards * backwards).mean(dim=0)

plt.imshow(F.relu(Variable(res)).data.numpy())








