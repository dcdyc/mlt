#Locally Weighted Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# w(x, x0) = e^((x-x0)^2/(-(2*k^2)))
def kernel(point, xmat, k):
	m, n = np.shape(xmat)
	weights = np.mat(np.eye((m))) #diagonal ones in the matrix
	for j in range(m):
		diff = point – X[j] #x-x0
		weights[j, j] = np.exp(diff*diff.T/(-2.0*k**2))
	return weights

def local_weight(point, xmat, ymat, k):
	wei = kernel(point, xmat, k)
	W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
	return W

def locally_weighted_regression(xmat, ymat, k):
	m, n = np.shape(xmat)
	ypred = np.zeros(m)
	for i in range(m):
		ypred[i] = xmat[i] * local_weight(xmat[i], xmat, ymat, k)
	return ypred

data = pd.read_csv(‘dataset.csv’)
bill = np.array(data.total_bill)
tip = np.array(data.tip)

mbill = np.mat(bill)
mtip = np.mat(tip)

m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T, mbill.T))

ypred = locally_weighted_regression(X, mtip, 0.5)

# Index sorting
sort_index = X[:, 1].argsort(0) #all rows but column index 1
xsort = X[sort_index][:, 0] #

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color=’green’)
ax.plot(xsort[:, 1], ypred[sort_index], color=’red’, linewidth=5)
plt.xlabel(‘Total bill’)
plt.ylabel(‘Tip’)
plt.show()
