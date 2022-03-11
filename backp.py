from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
iris = load_iris()
values = iris.data
target = iris.target
ss = StandardScaler()
values = ss.fit_transform(values)
x_train, x_test, y_train, y_test = train_test_split(values, target, test_size = 0.3, random_state =
1)
n = 1000
loss_cur = 999
classifier = MLPClassifier(hidden_layer_sizes = (4,3), activation = 'logistic', solver =
'sgd',learning_rate_init = 0.5, warm_start = True, random_state = 1,max_iter = 1, verbose =
True)
for _ in range(n):
classifier.fit(x_train, y_train)
loss_prev = loss_cur
loss_cur = classifier.loss_
for i in classifier.coefs_:
print(i, end='\n\n')
if(abs(loss_cur - loss_prev)<0.0001):
break
pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,pred)
#confusion_matrix
print("Confusion Matrix",cm,sep="\n")
#sep = "\n"
print("Accuracy", accuracy_score(y_test,pred))
for i in classifier.coefs_:
print(i)
