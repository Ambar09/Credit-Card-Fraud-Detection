#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 12:20:55 2019

@author: ambardubey
"""

# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

# Importing the dataset
dataset = pd.read_clipboard()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# To check null values
dataset.info() 
# To check correlation
corrMatt = dataset.corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=1, square=True,annot=True)
#Normalizing the dataset
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
#x,y =size of the grid
#input_len = number of features
#sigma = radius(default value)
#learning_rate decides by how much the weights has to be updated (default value)
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
#initialzing the weights
som.random_weights_init(X)
#epochs = 100
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
#window that will contain the map
bone()
#adding the values of MID in BMU
#taking transpose of MID matrix
pcolor(som.distance_map().T)
colorbar()

# Finding the fraudulent customers
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,3)], mappings[(8,4)]), axis = 0)
frauds = sc.inverse_transform(frauds)