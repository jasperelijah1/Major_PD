import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix 
#df = pd.read_csv('tremor - Copy.csv',encoding='ISO-8859-1',sep='delimiter')
df = pd.read_excel('jones_change_max.xlsx')
properties = list(df.columns)

X = df["prop_1"]
y = df['Activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train,X_test = np.array(X_train).reshape(-1,1),np.array(X_test).reshape(-1,1)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape,X_test.shape)

from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
test_acc = svm_model_linear.score(X_test, y_test) 
print('Test accuracy:', test_acc)

# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions)

