# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:01:42 2016

@author: shiyanlou
"""
# load datasets
from sklearn import datasets

iris = datasets.load_iris()
iris
iris.data.shape

import numpy as np 
np.unique(iris.target)

digits = datasets.load_digits()
digits
digits.data.shape
digits.images.shape
digits.target.shape
digits.target_names.shape
print(digits.data)
digits.data

# learning and predict
from sklearn import svm

clf = svm.SVC(gamma=0.001, C=100. )
clf.fit(digits.data[:-1],digits.target[:-1])
clf.predict(digits.data[-1])

# regression
from sklearn import linear_model

clf = linear_model.LinearRegression()
clf.fit([[0,0],[1,1],[2,2]],[0,1,1])
clf.coef_

#classifier:neighbors
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data,iris.target)
knn.predict([[0.1,0.2,0.3,0.4]])

#cluster:k-means
from sklearn import cluster, datasets

iris = datasets.load_iris()
k_means = cluster.KMeans(n_clusters=5)
k_means.fit(iris.data)
print k_means.labels_[::10] 
print iris.target[::10]

#image compress
from scipy import misc

lena = misc.lena().astype(np.float32)
X = lena.reshape(-1,1) 
k_means = cluster.KMeans(n_clusters=5)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
lena_compressed = np.choose(labels,values)
lena_compressed.shape = lena.shape

# diemnsionality reduction:PCA
from sklearn import decomposition

pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)

import pylab as pl 
X = pca.transform(iris.data)
pl.scatter(X[:,0],X[:,1],c=iris.target)




































