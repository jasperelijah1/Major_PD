from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
scaler = StandardScaler()


df = pd.read_excel('jones_dataset_max.xlsx')
inputs = df['Temperature']
target = df['Frequency']
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3, random_state=0, shuffle=True)
print(X_test, y_test)
X_train, X_test = np.array(X_train).reshape(-1, 1),np.array(X_test).reshape(-1, 1)
test_value = np.array([26.54455])
test_value = np.reshape(test_value, (-1, 1))
dtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
dtree_predictions = print(np.average(dtree_model.predict(test_value)))
print(dtree_model.score(X_test, y_test))

# creating a confusion matrix
# cm = confusion_matrix(y_test, dtree_predictions)
# print(cm)
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# scaler.fit(X_test)
# X_test = scaler.transform(X_test)
# test_value = np.array([10])
# test_value = np.reshape(test_value, (-1, 1))
# scaler.fit(test_value)
# test_value = scaler.transform(test_value)
# clf = svm.SVC()
# clf.fit(X_train, y_train)
# print(clf.predict(test_value))
# print(clf.score(X_test, y_test))

