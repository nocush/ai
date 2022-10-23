# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:20:19 2022

@author: matim
"""

#zadanie 2.5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes 
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = load_diabetes()
dane = pd.DataFrame(data.data, columns=data.feature_names)
dane['target']=data.target

korelacje=dane.corr() #macierz korelacji

X=dane.iloc[:,:dane.shape[1]-1]
y=dane.iloc[:,-1]

fig, ax= plt.subplots(X.shape[1],1,figsize=(2,20))
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col],y)
    
def testuj(n):
    s=0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
        linReg=LinearRegression()
        linReg.fit(X_train, y_train)
        y_pred=linReg.predict(X_test)
        s+=mean_absolute_percentage_error (y_test, y_pred)
    return s/n

def usun(n):
    s=0
    for i in range(n):
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2)
        outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
        X_train_no_outliers = X_train.loc[~outliers,:]
        y_train_no_outliers = y_train.loc[~outliers]
        linReg=LinearRegression()
        linReg.fit(X_train_no_outliers, y_train_no_outliers)
        y_pred=linReg.predict(X_test)
        s+=mean_absolute_percentage_error (y_test, y_pred)
    return s/n

def zamien(n):
    s=0
    for i in range(n):
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2)
        outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
        y_train_mean = y_train.copy()
        y_train_mean.loc[outliers] = y_train.mean()
        linReg=LinearRegression()
        linReg.fit(X_train, y_train_mean)
        y_pred=linReg.predict(X_test)
        s+=mean_absolute_percentage_error (y_test, y_pred)
    return s/n

testuj(12)
usun(10)
zamien(7)