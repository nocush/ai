from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout, GaussianNoise
from keras.layers import LayerNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.datasets import load_iris
import pandas as pd
data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)
y_one_hot = pd.get_dummies(y).values
class_num = y.shape[1]
neuron_num = 64
do_rate = 0.5
noise = 0.1
learning_rate = 0.001
block = [
 Dense,
 LayerNormalization(),
 BatchNormalization,
 Dropout,
 GaussianNoise]
args = [
 (neuron_num,'selu'),(),(),(do_rate,),(noise,)]
model = Sequential()
model.add(Dense(neuron_num, activation='relu', input_shape = (X.shape[1],)))
repeat_num = 2
for i in range(repeat_num):
    for layer,arg in zip(block, args):
        model.add(layer(*arg))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer= Adam(learning_rate),
              loss='binary_crossentropy',
              metrics=('accuracy', 'Recall', 'Precision'))

model.fit(X_train, y_train, batch_size=32,
          epochs=nrEpoch[0], validation_data=(X_test, y_test),
          verbose=0)
acc=max(model.history.history['val_accuracy'])
