# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 18:23:55 2018

@author: Harry
"""

#Polynomial Regression
#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset

dataset = pd.read_csv('Position_Salaries.csv')
#We use 1:2 in order to create X as a matrix not a vector
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

#Fitting polynomial model

#The regressor in this model will create x^2, x^3,...,x^n
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#We use fit_transform in order to tranform the X to its
#polynomial counterpart if we need to work in a polynomial space
plrReg = PolynomialFeatures(degree=4)
X_poly = plrReg.fit_transform(X)
#plrReg acts as a tool to create the polynomial X space
#We fit the mdoel like a normal slr btu with a polynomial X
plrModel = LinearRegression().fit(X_poly,y)

#Predicting a new result with Polynomial Regression

plrModel.predict(6.5)

#Visualizing the Polynomial Linear Regression

#X_grid is used to smooth the lines and reshape to make it a matrix
X_grid = np.arange(min(X), max(X),0.1).reshape()
plt.scatter(X_grid, y, color = 'red')
plt.plot(X, plrModel.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()