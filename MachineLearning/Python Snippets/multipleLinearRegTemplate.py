# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 18:19:19 2018

@author: Harry
"""
#Multiple Linear Regression Template
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv('50_Startups.csv')
dataset.head()
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Test for model requirements
# 1) Linearity
# 2) Homoscedasticity
# 3) Correlation in error terms
# 4) Colinearity
# 5) Outliers and leverage points

#Encoding categorical vars
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,3] = LabelEncoder().fit_transform(X[:,3])
X = OneHotEncoder(categorical_features=[3]).fit_transform(X).toarray()

#Avoiding the Dummy var trap (-1 var)
X = X[:,1:]

#Split to train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=0)

#Fiting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict the test set results
y_pred = regressor.predict(X_test)
y_hat = y_pred - y_test

#Building the optimal model with Backward Elimination
#Custom code for automatic elimination
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_ols = sm.OLS(y,x).fit()
        maxVar = max(regressor_ols.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars -i):
                if (regressor_ols.pvalues[j].astype(float) == maxVar):
                    
