# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:18:33 2022

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

data = pd.read_excel("zadanie_1.xlsx")

#metoda zmieniajace wartosci tekstowe na wartosci liczbowe
def qualitative_to_0_1(data, column, value_to_be_1):
    
    columns = list(data.columns)
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

data = qualitative_to_0_1(data, "Gender", "Female")
data = qualitative_to_0_1(data, "Married", "Yes")
data = qualitative_to_0_1(data, "Education", "Graduate")
data = qualitative_to_0_1(data, "Self_Employed", "Yes")
data = qualitative_to_0_1(data, "Loan_Status", "Y")

cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis = 1)
data = data.drop(columns = ["Property_Area"])

#podzial na podzbiory
features = data.columns
vals = data.values.astype(np.float)
y = data["Loan_Status"].astype(np.float)
X = data.drop(columns=["Loan_Status"]).values.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

models = [kNN(n_neighbors=3, weights="uniform"),kNN(n_neighbors=5, weights="distance")]
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred) #zmienna do przechwytywania macierzy poylek
    print(cm)


