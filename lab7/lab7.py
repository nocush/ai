# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:28:19 2022

@author: mat
"""

from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
train, test = mnist.load_data()
X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
class_cnt = np.unique(y_train).shape[0]
filter_cnt = 32
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu'
kernel_size = (3,3)
model = Sequential()
conv_rule = 'same'
model.add(Conv2D(input_shape = X_train.shape[1:],
 filters=filter_cnt,
 kernel_size = kernel_size,
 padding = conv_rule, activation = act_func))
model.add(Flatten())
model.add(Dense(class_cnt, activation='softmax'))
model.compile(optimizer=Adam(learning_rate),
 loss='SparseCategoricalCrossentropy',
 metrics='accuracy')
model.fit(x = X_train, y = y_train,
 epochs = class_cnt ,
 validation_data=(X_test, y_test))

#7.3
from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, MaxPooling2D
filter_cnt = 32
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu'
kernel_size = (3,3)
pooling_size = (2,2)
model = Sequential()
conv_rule = 'same'
model.add(Conv2D(input_shape = X_train.shape[1:],
 filters=filter_cnt,
 kernel_size = kernel_size,
 padding = conv_rule, activation = act_func))
model.add(MaxPooling2D(pooling_size))
model.add(Flatten())
model.add(Dense(class_cnt, activation='softmax'))
model.compile(optimizer=Adam(learning_rate),
loss='SparseCategoricalCrossentropy',
 metrics='accuracy')

#7.4
from keras.datasets import cifar10
(x_train, y_train),(x_test,y_test) = cifar10.load_data()
class_cnt = np.unique(y_train).shape[0]
filter_cnt = 32
neuron_cnt = 64
learning_rate = 0.0001
act_func = 'relu'
kernel_size = (3,3)
pooling_size = (2,2)
class_cnt = np.unique(y_train).shape[0]
model = Sequential()
conv_rule = 'same'
model.add(Conv2D(input_shape = x_train.shape[1:],
 filters=filter_cnt,
 kernel_size = kernel_size,
 padding = conv_rule, activation = act_func))
model.add(MaxPooling2D(pooling_size))
model.add(Flatten())
model.add(Dense(class_cnt, activation='softmax'))
model.compile(optimizer=Adam(learning_rate),
loss='SparseCategoricalCrossentropy',
 metrics='accuracy')
model.fit(x = x_train, y = y_train,
 epochs = 12 ,
 validation_data=(x_test, y_test))

