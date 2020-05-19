---
title: "Cats vs Dogs"
date: 2020-05-19
tags: [Deep learning - Neural Networks]
excerpt: "A network that can differentiate between images of cats and dogs"
---

## Cats vs Dogs - Neural Networks

I've created a neural network with 4 hidden layers which take 1 channel input and give 2 outputs (Cat / Dog).
(Used convolution for 3 hidden layers and linear for 1 hidden layer)

Before reading the code check if you have installed Pytorch . If not consider checking out my guide on it's installation [here](https://nvikramraj.github.io/Anaconda/)


## The steps involved in making this neural network :

* Getting the dataset (input)
* Establish the neural network
* Train the neural network
* Validate the neural network
* Plotting graphs
* Test the neural network

**Packages to be imported**

```python

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

```

Activating the GPU :

```python

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")

else:
    device = torch.device("cpu")
    print("running on the CPU")

# If gpu is avaliable it will do calculations on the GPU

```

## Getting the dataset :

I've used the dataset from microsoft . [Download](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
Extract the folder "PetImages" to your code folder.

Now , you need to convert the images in those folder to an array.
In this code I've converted the image to grayscale and resized it to 50x50 pixels.
Then used np.array to convert the image to a matrix.

**Code**

```python

#make sure to extract kaggle PetTmages folder in the same file as your code

REBUILD_DATA = True # True - to build a dataset , false - to not build a dataset

class DogsVSCats():
    IMG_SIZE = 50 # making the img 50x50 pixels
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)): #runs till the last img in the directory
                if "jpg" in f: #getting only jpg
                    try: #used for error handling because there are some corrupt imgs
                        path = os.path.join(label, f) 
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)#converting to gray scale to reduce complexity
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE)) 
                        # converts the img to 50x50 px
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]]) 
                        # assigns [1,0] as cat and [0,1] as dog (using one hot vector) 

                        if label == self.CATS: #used to check balance of inputs b/w cats and dogs
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1
                    except Exception as e:
                        pass

        np.random.shuffle(self.training_data) #shuffling the cats and dogs data for efficient generalization
        np.save("training_data.npy",self.training_data) #saving it
        print("Cats :",self.catcount) #checking the count
        print("Dogs :",self.dogcount)

if REBUILD_DATA: #To build the data REBUILD_DATA = True
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()


```

The data given should be balanced . Imagine in this case we got two types of inputs cats and dogs , So the data should contain 50% of cat images and 50% of dog images . 

Try avoiding data that is not balanced . It would cause the neural network to memorize than generalize.
After creating the data set you can either delete the code or change REBUILD_DATA = False on the top.


## Creating the neural network :

This neural network contains 4 hidden layers , 3 2-d convolution layers and 1 linear layer .

The first layer gets input from 1 channel and has 32 neurons.
The second layer gets input from first layer and has 64 neurons.
The third layer gets input from second layer and has 128 neurons.

To pass data from the convolution layer to the linear layer , we need to flatten the image.
So I've created a random matrix of 50x50 size to find the shape of data after it passes the convolution layer.
The linear layer has 512 neurons and gives the output as a probability of Cats vs Dogs.

Activation function used : Rectified Linear 
Activation function used on output : softmax


```python

class Net(nn.Module):
    def __init__(self):
        super().__init__() #creating hidden layers
        #no of channels used is 1
        # 3 hidden layers with 32 / 64 / 128 neurons . this caculates in convolution 2d
        self.conv1 = nn.Conv2d(1,32,5) # 1 channel , 32 neurons , 5 - kernel size
        self.conv2 = nn.Conv2d(32,64,5) 
        self.conv3 = nn.Conv2d(64,128,5)

        x = torch.randn(50,50).view(-1,1,50,50) # we need to convert 2d to 1d so we need to find the 1d size
        self._to_linear = None # using random generated data to find the size
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear,512) # hidden layer with 512 neurons  
        self.fc2 = nn.Linear(512,2) # out put 2 neurons cat / dog

    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))#using pooling and activation function to round off values
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))

        #print(x[0].shape)

        if self._to_linear is None:

            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] # to get the size of 1-d or flattened img

        return x

    def forward (self,x): #used to pass through the hidden layers
        x = self.convs(x) # calculating convolution first
        x = x.view(-1,self._to_linear) # converting to linear
        x = F.relu(self.fc1(x)) # calculating linear
        x = self.fc2(x) # getting output
        return F.softmax(x,dim =1) #using activation function at output to get % or 0-1 values

```