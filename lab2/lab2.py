# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:23:50 2022

@author: matim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_excel("practice_lab_2.xlsx")
korelacje = data.corr()

X = data.iloc[:,:data.shape[1]-1]
y = data.iloc[:,-1]

#zad. 2.1
fig, ax = plt.subplots(X.shape[1],1,figsize=(5,20))
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col],y)
    
#zad 2.2

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def testuj_model(n):
    sum = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
        random_state=221, shuffle=True)
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        mape = mean_absolute_percentage_error (y_test, y_pred)
        sum += mape
    return sum/n


testuj_model(10)

#zad 2.3
def usuniecie(n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
        random_state=221, shuffle=True)
        outliers = np.abs((y_train - y_train.mean())/ y_train.std())>3
        X_train_no_outliers = X_train[~outliers,:]
        y_train_no_outliers = y_train[~outliers]
        linReg = LinearRegression()
        linReg.fit(X_train_no_outliers,y_train_no_outliers)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error (y_test, y_pred)
    return s/n

usuniecie(10)

def zamiana(n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
        random_state=221, shuffle=True)
        outliers = np.abs((y_train - y_train.mean())/ y_train.std())>3
        y_train_mean = y_train.copy()
        y_train_mean[outliers] = y_train.mean()
        linReg = LinearRegression()
        linReg.fit(X_train,y_train_mean)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error (y_test, y_pred)
    return s/n

zamiana(10)

