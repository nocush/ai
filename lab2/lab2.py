# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:34:35 2022

@author: JB
"""
#do niedzieli 2.5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plik = pd.read_excel("practice2.xlsx")
korelacje = plik.corr()

X = plik.iloc[:,:plik.shape[1]-1]
y = plik.iloc[:,-1]

#z2.1
fig, ax = plt.subplots(X.shape[1],1,figsize=(5,20))
for i, col in enumerate(X.columns):
        ax[i].scatter(X[col],y)


#z2.2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,  mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

def testuj_model(n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error (y_test, y_pred)  
    return s/n

testuj_model(10)

#z2.3
def usuniecie(n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
        outliers = np.abs((y_train - y_train.mean())/ y_train.std())>3
        X_train_no_outliers = X_train.loc[~outliers,:]
        y_train_no_outliers = y_train.loc[~outliers]
        linReg = LinearRegression()
        linReg.fit(X_train_no_outliers, y_train_no_outliers)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error (y_test, y_pred)  
    return s/n
usuniecie(10)

def zmiana(n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
        outliers = np.abs((y_train - y_train.mean())/ y_train.std())>3
        y_train_mean = y_train.copy()
        y_train_mean[outliers] = y_train.mean()
        linReg = LinearRegression()
        linReg.fit(X_train, y_train_mean)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error (y_test, y_pred)  
    return s/n
zmiana(10)

#2.5
from sklearn.datasets import load_diabetes
data = load_diabetes()
dane = pd.DataFrame(data.data, columns=data.feature_names)
dane['target']=data.target
korelacje = dane.corr()




