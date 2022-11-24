## Import Libraries and Packages
import os
import sys
import tensorflow as tf
import keras
import platform
import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros, ones
from numpy.random import randn, randint
import cv2
from keras.layers import Dense, Reshape, Flatten, LeakyReLU, Conv2DTranspose, Conv2D, Dropout, Embedding, Concatenate, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn import preprocessing

## Versions 
print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"Python {sys.version}")

## Set Artists Names for 10 Artists, with 100 Arts per each
list = ['Alfred_Sisley',
'Amedeo_Modigliani', 
'Gustav_Klimt',
'Marc_Chagall',
'Pablo_Picasso',
'Paul_Klee',
'Peter_Paul_Rubens',
'Pieter_Bruegel',
'Raphael',
'Rembrandt']
print(f"Number of Artists: {len(list)}")

## Import dataset with Arts and Artist Names
print("Importing the Dataset...")
directory = '/Users/Sachith/dataset/Arts'
categories = list

directory = directory
categories = categories

data = []
labels = []
        
for category in categories:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        data.append(image)
        labels.append(category)
        
n_arts = f"Number of Arts: {len(data)} loaded"
n_artists = f"Number of Artists: {len(list)}"

print(n_arts)
print(n_artists)
print("Loading Completed!!")

