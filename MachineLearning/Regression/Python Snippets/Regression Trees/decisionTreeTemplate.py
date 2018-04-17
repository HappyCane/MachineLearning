# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 22:50:54 2018

@author: Harry
"""

#Decision Tree Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset

dataset = pd.read_csv("Position_Salaries.csv")
#We use 1:2 in order to create X as a matrix not a vector
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Fitting the tree

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#Making predictions (we scale 6.5 as well ~ transform
#require an array as input)

y_pred = regressor.predict(6.5)

#Visualizing Decision Tree Regression
#X_grid is used to smooth the lines and reshape to make it a matrix
X_grid = np.arange(min(X), max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff (Decision Tree)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()