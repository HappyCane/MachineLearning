# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:33:06 2018

@author: Harry
"""
#Simple Linear Regression Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Split to train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=1/3, random_state=0)

#Fit the linear regression model
#No need to scale because to LinearReg lib will do it
#We create a regressor object
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting test set results
#Enter the test set in predict
#The predicted salaries for the TEST set
y_pred = regressor.predict(X_test)

#Visuzalize the training set results
plt.scatter(X_train, y_train, color = 'red')
#We dont use y_pred because we want the predictions for
#the TRAIN set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visuzalize the test set results
plt.scatter(X_test, y_test, color = 'red')
#We dont use y_pred because we want the predictions for
#the TRAIN set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
