# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d_fUVn_DXQsQkcznfp1ywjqBAUV0QqqL
"""

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import EarlyStopping
#df = pd.read_csv('tremor - Copy.csv',encoding='ISO-8859-1',sep='delimiter')
df = pd.read_excel('jones_dataset_max.xlsx')#,encoding='ISO-8859-1',engine='c')#,sep='delimiter')
properties = list(df.columns)



X = df["prop_1"]
y = df['Activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train,X_test = np.array(X_train).reshape(-1,1),np.array(X_test).reshape(-1,1)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape,X_test.shape)
model = keras.Sequential([
         keras.layers.Input(shape = (1,)),                
    keras.layers.Dense(10, activation=tf.nn.relu),
	keras.layers.Dense(5, activation=tf.nn.relu),
  
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])
es = EarlyStopping(monitor='accuracy', mode='max',patience=3)
sgd = keras.optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.87, nesterov=True)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

model.fit(X_train, y_train, epochs=50, batch_size=5,callbacks=[es])

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

