# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("zadanie.csv")
col = list(data)
val = data.values

mean_col = val.mean(axis = 0)
mean_std = val.std()
difference = val - mean_std
max_row_val = val.max(axis = 1)
arr2 = val*2
col_max = val.max(axis = 0)
col2 = np.array(col)
zadh = col2[col_max == val.argmax(axis = 0)]
arr9 = val.sum(axis = 0)


#zadanie 2
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

plik = pd.read_csv("zadanie.csv")
korelacje = plik.corr()
X = plik.iloc[:,plik.shape[1]-1]
y = plik.iloc[:,-1]
fig, ax = plt.subplots(X.shape[0],1,figsize=(1,20))
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col],y)
    

