#K Nearest Neighbor Classification

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

datas = load_iris()

x=datas.data
y=datas.target

x_train , x_test ,y_train ,y_test = train_test_split(x,y,test_size = 0.4,random_state=9)

model = KNeighborsClassifier(n_neighbors = 7)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print('Accuracy: ',accuracy_score(y_test,y_pred))

predValue = []
# MisSamplified Counts
pd.DataFrame({'Actual':y_test,'prediction':y_pred})
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        predValue.append(0)
    else:
        predValue.append(-1)
misSamplified = predValue.count(-1)
print("MisSamplified Samples Count: ", misSamplified)