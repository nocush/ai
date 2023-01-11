# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:47:29 2022

@author: matim
"""

from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D,Dense, Input, Reshape, UpSampling2D, BatchNormalization, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_train_scaled = (x_train/255).copy()

act_func = 'relu'
aec_dim_num = 2
encoder_layers = [
    Dense(32,activation=act_func),
    Dense(32,activation=act_func),
    Dense(32,activation=act_func),
    Dense(32,activation=act_func)
                  ]
decoder_layers = [
    Dense(32,activation=act_func),
    Dense(32,activation=act_func),
    Dense(32,activation=act_func),
    Dense(32,activation=act_func)]