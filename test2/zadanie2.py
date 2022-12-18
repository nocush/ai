# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:00:26 2022

@author: mat
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.pipeline import Pipeline

data = pd.read_csv("zadanie_2.csv")

#metoda zmieniajace wartosci tekstowe na wartosci liczbowe
def qualitative_to_0_1(data, column, value_to_be_1):
    
    columns = list(data.columns)
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data


data = qualitative_to_0_1(data, "label", "female")

#podzial na podzbiory
features = list(data.columns)
vals = data.values.astype(np.float)
X = vals[:,:-1]
y = vals[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#wykres
X_paced = PCA(2).fit_transform(X_train) #2 główne składowe
fig,ax = plt.subplots(1,1)
females = y_train == 1
ax.scatter(X_paced[females,0],X_paced[females,1],label="female")
ax.scatter(X_paced[~females,0],X_paced[~females,1],label="male")
ax.legend()
