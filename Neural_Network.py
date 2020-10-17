#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from numpy.random import randint
from numpy.random import random as rnd
from random import gauss,randrange
from numpy import cumsum
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
df = pd.read_excel("up.xlsx",sep='\\s*,\\s*')
X=df.iloc[:,:5]
Y=df.iloc[:,5:]
#Preprocessing the entire set
scaler = preprocessing.StandardScaler().fit(X)
scalery = preprocessing.StandardScaler().fit(Y)
yn=scalery.transform(Y) #Normalizing test labels
xtrain,xtest,ytrain,ytest = train_test_split(X,yn)
xtrain1 = scaler.transform(xtrain)
xtest1 = scaler.transform(xtest)
#Developing the model
def build_model():
	model = keras.models.Sequential([keras.layers.Dense(4,tf.nn.relu,input_dim=4),keras.layers.Dense(3,tf.nn.relu),keras.layers.Dense(4)])
	return model
model=build_model()
model.summary()
EPOCHS=576
optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mean_absolute_error'])
history = model.fit(xtrain1, ytrain,epochs=EPOCHS,validation_split = 0.2, verbose=2) #Trains the model
model.save('newmod1.h5') #Saving the entire trained model as newmod1.h5 - ref Genetic_Algorithm_Optimization.py for optimization using this model