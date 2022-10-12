import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data = pd.read_excel(r"D:/GitHub/ai/lab1/practice_lab_1.xlsx")
cols = list(data)
vals = data.values
#zad 1.2.1
arr1 = vals[::2,:]
arr2 = vals[1::2,:]
diff = arr1 - arr2
#ex 1.2.2
avg = vals.mean()
sr = vals.std()
arr3 = (vals - avg)/sr
#ex 1.2.3
avg2 = vals.mean(axis = 0)
diff2 = vals.std(axis = 0)
arr4 = (vals - avg2)/(diff2 + np.spacing(diff2))
#ex 1.2.4
arr5 = diff2/(avg2 + np.spacing(diff2))
#ex 1.2.5
maxvalue = np.argmax(arr5)
#ex 1.2.6
zad6 = (vals>vals.mean(axis=0)).sum(axis=0)
#ex 1.2.7
max_value = vals.max()
col_max = vals.max(axis=0)
cols2 = np.array(cols)
cols2[col_max == max_value]

#zadanie 1.3
x = np.arange(-5,5,0.01)
y = np.tanh(x)
wyk1 = plt.plot(x,y)

#
plt.plot(x[x>0],x[x>0])
plt.plot(x[x<=0],np.exp(x[x<=0])-1)