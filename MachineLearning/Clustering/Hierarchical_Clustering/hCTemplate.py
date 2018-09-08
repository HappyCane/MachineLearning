# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 07:05:33 2018

@author: Harry
"""

#Hierarchical Clustering

#Importing libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values #Don't forget to convert to values!

#Using the dendrograms to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendro = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

#Fitting HC to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean',
                             linkage='ward')
y_hc = hc.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[y_hc == 0,0], X[y_hc==0,1], s=100,
            c='red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1,0], X[y_hc==1,1], s=100,
            c='blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2,0], X[y_hc==2,1], s=100,
            c='green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3,0], X[y_hc==3,1], s=100,
            c='cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4,0], X[y_hc==4,1], s=100,
            c='magenta', label = 'Cluster 5')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score (1-100)')
plt.legend
plt.show()