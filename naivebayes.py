#NaiveBayes Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataset.csv")
data = data.apply(lambda x : pd.factorize(x)[0]) # labels the data with 0, 1, 2, 3,...

x = np.array(data.iloc[:,0:-1]) # except the last column
y = np.array(data.iloc[:,-1]) # only the last column
x_train , x_test ,y_train ,y_test = train_test_split(x,y,test_size = 0.4,random_state=1)

model = GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))