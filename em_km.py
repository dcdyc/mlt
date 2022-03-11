from sklearn.metrics import completeness_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
values = iris.data
target = iris.target

plt.figure(figsize=(14, 7))
colormap = np.array(['red','lime','black'])

plt.subplot(1, 3, 1)
plt.scatter(values[:,2], values[:,3], c= colormap[target]) #colormap[target]
plt.title("Actual Plot")

#KMeans
model = KMeans(n_clusters = 3) #n_clusters = 3
model.fit(values) #target
plt.subplot(1, 3, 2)
plt.title("K-Means")
plt.scatter(values[:,2], values[:,3], c = colormap[model.labels_])
print("Completeness Score", completeness_score(target, model.labels_))

#GaussianMixture
model = GaussianMixture(n_components = 3)
model.fit(values)
pred = model.predict(values)
plt.subplot(1, 3, 3)
plt.title("GMM")
plt.scatter(values[:,2],values[:,3],c=colormap[pred])
print("Completeness_score", completeness_score(target, pred))