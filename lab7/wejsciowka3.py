# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:05:25 2022

@author: mat
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
import tensorflow as tf

from keras.datasets import fashion_mnist
train, test = tf.keras.datasets.fashion_mnist.load_data()

X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
class_cnt = np.unique(y_train).shape[0]

filter_cnt = 32
units = 32
learning_rate = 0.0001
kernel_size=(3,3)
pooling_size=(2,2)
conv_rule='same'

model=Sequential()
model.add(layer=Conv2D(input_shape = X_train.shape[1:],filters=filter_cnt,kernel_size=kernel_size,padding=conv_rule))
model.add(layer=MaxPooling2D(pooling_size))
model.add(layer=Conv2D(filter_cnt, kernel_size=kernel_size))
model.add(layer=MaxPooling2D(pooling_size))
model.add(layer=Flatten())
model.add(layer=Dense(class_cnt))
model.add(layer=Dense(class_cnt,activation='softmax'))

model.compile(optimizer=Adam(learning_rate),loss='SparseCategoricalCrossentropy',metrics='acc')

history=model.fit(X_train,y_train,batch_size=32,epochs=5,validation_data=(X_test,y_test))

floss_train = history.history['loss']
floss_test = history.history['val_loss']
acc_train = history.history['acc']
acc_test = history.history['val_acc']
fig,ax = plt.subplots(1,2,figsize=(20,10))
epochs = np.arange(0,5)
ax[0].plot(epochs,floss_train,label='floss_train')
ax[0].plot(epochs,floss_test,label='floss_test')
ax[0].set_title('Funkcja strat')
ax[0].legend()
ax[1].plot(epochs,acc_train,label='acc_train')
ax[1].plot(epochs,acc_test,label='acc_test')
ax[1].set_title('Dokladnosci')
ax[1].legend()

