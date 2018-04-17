# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:28:13 2018

@author: Harry
"""

#SVR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset

dataset = pd.read_csv("Position_Salaries.csv")
#We use 1:2 in order to create X as a matrix not a vector
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Feature Scaling
#The SVR class does not contain scaling so we need
#to do it manually

from sklearn.preprocessing import StandardScaler
#If u have initiated the object u only need to fit once
#Otherwise you have to fit_transform for both sets
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
#We need to reshape y into an int64 array for the scaler
#to work
y = np.reshape(y, (len(X),1))
y = sc_y.fit_transform(y)
y = np.reshape(y, (len(X),))

#Fitting SVR to the dataset

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#Making predictions (we scale 6.5 as well ~ transform
#require an array as input)

y_pred = sc_y.inverse_transform(
        regressor.predict(sc_X.transform(
                np.reshape(6.5,(1,-1)))))

#Visualizing SVR
#X_grid is used to smooth the lines and reshape to make it a matrix

X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()